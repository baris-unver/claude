#include "vio_hold/vh_hold.h"
#include "vio_hold/vh_fast.h"
#include "vio_hold/vh_klt.h"

#include <math.h>
#include <string.h>

/* Median via insertion sort; n is at most VH_MAX_FEATURES. */
static float median(float *v, int n)
{
    for (int i = 1; i < n; i++) {
        const float key = v[i];
        int j = i - 1;
        while (j >= 0 && v[j] > key) {
            v[j + 1] = v[j];
            j--;
        }
        v[j + 1] = key;
    }
    return (n & 1) ? v[n / 2] : 0.5f * (v[n / 2 - 1] + v[n / 2]);
}

void vh_init(vh_ctx *ctx, const vh_params *prm)
{
    memset(ctx, 0, sizeof(*ctx));
    ctx->prm = *prm;
    vh_rot_init(&ctx->rot, prm->r_cb);
}

void vh_gyro(vh_ctx *ctx, uint64_t t_us, float wx, float wy, float wz)
{
    vh_rot_push(&ctx->rot, t_us, wx, wy, wz);
}

int vh_set_keyframe(vh_ctx *ctx, const vh_image *img, uint64_t t_us)
{
    vh_corner corners[VH_MAX_FEATURES];
    int n = vh_fast_detect(img, corners);
#if VH_FAST_THRESHOLD_LO < VH_FAST_THRESHOLD
    /* Low-contrast scene (fog, precipitation): one relaxed retry. Truly
     * featureless scenes still fail this and the keyframe stays rejected. */
    if (n < VH_MIN_TRACKED)
        n = vh_fast_detect_ex(img, corners,
                              VH_FAST_THRESHOLD_LO, VH_FAST_MIN_SCORE_LO);
#endif
    if (n < VH_MIN_TRACKED)
        return 0;

    /* Copy the frame so the keyframe template outlives the caller's buffer. */
    for (int y = 0; y < VH_IMG_H; y++)
        memcpy(ctx->key_img + (int32_t)y * VH_IMG_W,
               img->data + (int32_t)y * img->stride, VH_IMG_W);

    vh_image key = { ctx->key_img, VH_IMG_W, VH_IMG_H, VH_IMG_W };
    vh_pyr_build(&ctx->key_pyr, &key, ctx->key_store);

    for (int i = 0; i < n; i++) {
        ctx->fx[i] = corners[i].x;
        ctx->fy[i] = corners[i].y;
        ctx->bearing[i] =
            vh_cam_pixel_to_bearing(&ctx->prm.cam, corners[i].x, corners[i].y);
        ctx->active[i] = true;
    }
    ctx->n_key = n;
    ctx->key_t_us = t_us;
    ctx->have_key = true;
    ctx->last_res_x = ctx->last_res_y = 0.0f;

    vh_rot_rekey(&ctx->rot, t_us);
    return n;
}

vh_result vh_process_frame(vh_ctx *ctx, const vh_image *img, uint64_t t_us)
{
    vh_result r;
    memset(&r, 0, sizeof(r));
    r.t_us = t_us;

    if (!ctx->have_key) {
        r.status = VH_STATUS_NO_KEYFRAME;
        if (ctx->prm.auto_rekey && vh_set_keyframe(ctx, img, t_us) > 0)
            r.rekeyed = true;
        r.key_t_us = ctx->key_t_us;
        r.n_keyframe = ctx->n_key;
        return r;
    }

    vh_rot_integrate_to(&ctx->rot, t_us);
    vh_pyr_build(&ctx->cur_pyr, img, ctx->cur_store);

    float rx[VH_MAX_FEATURES], ry[VH_MAX_FEATURES], rdiv[VH_MAX_FEATURES];
    int n_ok = 0;

    for (int i = 0; i < ctx->n_key; i++) {
#ifdef VH_DEBUG_TRACKS
        ctx->dbg_state[i] = 0;
#endif
        if (!ctx->active[i])
            continue;

        /* Predict pixel under pure rotation since the keyframe. */
        const vh_vec3 bc = vh_rot_key_to_cur(&ctx->rot, ctx->bearing[i]);
        float pu, pv;
        if (!vh_cam_bearing_to_pixel(&ctx->prm.cam, bc, &pu, &pv))
            continue; /* rotated behind the image plane; try again next frame */

        /* Warm-start with last frame's median translation residual. */
        float cu = pu + ctx->last_res_x;
        float cv = pv + ctx->last_res_y;
#ifdef VH_DEBUG_TRACKS
        ctx->dbg_state[i] = 1;
        ctx->dbg_pred[i][0] = pu;
        ctx->dbg_pred[i][1] = pv;
#endif
        if (!vh_klt_track(&ctx->key_pyr, &ctx->cur_pyr,
                          ctx->fx[i], ctx->fy[i], &cu, &cv))
            continue;
#ifdef VH_DEBUG_TRACKS
        ctx->dbg_state[i] = 2;
        ctx->dbg_trk[i][0] = cu;
        ctx->dbg_trk[i][1] = cv;
#endif

        const float dx = cu - pu;
        const float dy = cv - pv;
        rx[n_ok] = dx;
        ry[n_ok] = dy;

        /* Radial component about the principal point measures looming:
         * scene expansion (positive) = forward motion since the keyframe. */
        const float ax = pu - ctx->prm.cam.cx;
        const float ay = pv - ctx->prm.cam.cy;
        const float an = sqrtf(ax * ax + ay * ay);
        rdiv[n_ok] = (an > 20.0f) ? (dx * ax + dy * ay) / an : 0.0f;

        n_ok++;
    }

    r.n_keyframe = ctx->n_key;
    r.n_tracked = n_ok;
    r.key_t_us = ctx->key_t_us;

    if (n_ok >= VH_MIN_TRACKED) {
        /* median() sorts its input, so give each call its own copy. */
        float tmp[VH_MAX_FEATURES];
        memcpy(tmp, rx, sizeof(float) * (size_t)n_ok);
        r.res_x_px = median(tmp, n_ok);
        memcpy(tmp, ry, sizeof(float) * (size_t)n_ok);
        r.res_y_px = median(tmp, n_ok);
        memcpy(tmp, rdiv, sizeof(float) * (size_t)n_ok);
        r.divergence_px = median(tmp, n_ok);

        ctx->last_res_x = r.res_x_px;
        ctx->last_res_y = r.res_y_px;

        const float frac = (float)n_ok / (float)ctx->n_key;
        r.status = (frac < VH_REKEY_FRACTION) ? VH_STATUS_DEGRADED : VH_STATUS_OK;
    } else {
        r.status = VH_STATUS_LOST;
        ctx->last_res_x = ctx->last_res_y = 0.0f;
        if (ctx->prm.auto_rekey && vh_set_keyframe(ctx, img, t_us) > 0)
            r.rekeyed = true;
    }
    return r;
}
