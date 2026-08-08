/*
 * test_real_texture.c — hover-regime tests on REAL imagery.
 *
 * Loads one real frame from a preprocessed DCS sequence (.vhr, see
 * tools/dcs_extract.py) and uses it as the scene: known sub-pixel
 * translations and gyro-fed rotations are injected by warping the real
 * frame, so the pipeline runs on real texture statistics with exact ground
 * truth. Complements tests/test_pipeline.c (synthetic texture) and
 * tools/replay_dcs.c (real motion, no ground truth).
 *
 * Covers what the forward-flight replays cannot:
 *   - hover-magnitude excursions around a fixed keyframe
 *   - rotation compensation at rates the archive lacks (>> 7 deg/s)
 *   - drift-freeness over a multi-hundred-frame hover with no re-key
 *
 * Usage: test_real_texture <file.vhr> [frame_index]
 */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "vio_hold/vh_hold.h"
#include "vio_hold/vh_fast.h"
#include "vio_hold/vh_bias.h"

#define W VH_IMG_W
#define H VH_IMG_H

static uint8_t g_base[W * H];
static uint8_t g_cur[W * H];
static int g_failures = 0;

#define CHECK(cond, ...)                                                     \
    do {                                                                     \
        if (cond) {                                                          \
            printf("  PASS: " __VA_ARGS__);                                  \
            printf("\n");                                                    \
        } else {                                                             \
            printf("  FAIL: " __VA_ARGS__);                                  \
            printf("\n");                                                    \
            g_failures++;                                                    \
        }                                                                    \
    } while (0)

/* ---------- .vhr loading ---------- */

static uint16_t rd_u16(FILE *f) { uint8_t b[2] = {0}; if (fread(b, 1, 2, f) != 2) {} return (uint16_t)(b[0] | b[1] << 8); }
static uint32_t rd_u32(FILE *f) { uint8_t b[4] = {0}; if (fread(b, 1, 4, f) != 4) {} return (uint32_t)b[0] | (uint32_t)b[1] << 8 | (uint32_t)b[2] << 16 | (uint32_t)b[3] << 24; }
static float    rd_f32(FILE *f) { uint32_t u = rd_u32(f); float v; memcpy(&v, &u, 4); return v; }

/* Loads frame `idx` into g_base and returns the file's camera. */
static vh_camera load_vhr_frame(const char *path, uint32_t idx)
{
    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); exit(2); }
    if (rd_u32(f) != 0x56484452u || rd_u32(f) != 1) {
        fprintf(stderr, "%s: bad header\n", path);
        exit(2);
    }
    uint16_t w = rd_u16(f), h = rd_u16(f);
    vh_camera cam;
    memset(&cam, 0, sizeof cam);
    cam.fx = rd_f32(f); cam.fy = rd_f32(f);
    cam.cx = rd_f32(f); cam.cy = rd_f32(f);
    (void)rd_f32(f); /* tilt */
    uint32_t n_gyro = rd_u32(f), n_frames = rd_u32(f);
    if (w != W || h != H) {
        fprintf(stderr, "%s: %ux%u but library built for %ux%u\n", path, w, h, W, H);
        exit(2);
    }
    if (idx >= n_frames) {
        fprintf(stderr, "%s: frame %u out of range (%u frames)\n", path, idx, n_frames);
        exit(2);
    }
    if (fseek(f, (long)(n_gyro * 20u) + (long)idx * (long)(8u + W * H) + 8, SEEK_CUR) != 0) {
        perror("fseek");
        exit(2);
    }
    if (fread(g_base, 1, W * H, f) != W * H) {
        fprintf(stderr, "%s: truncated\n", path);
        exit(2);
    }
    fclose(f);
    return cam;
}

/* ---------- warp (test-side ground truth), as in test_pipeline.c ---------- */

static float sample_clamped(const uint8_t *img, float x, float y)
{
    if (x < 0.0f) x = 0.0f;
    if (y < 0.0f) y = 0.0f;
    if (x > (float)(W - 2)) x = (float)(W - 2);
    if (y > (float)(H - 2)) y = (float)(H - 2);
    const int ix = (int)x, iy = (int)y;
    const float ax = x - (float)ix, ay = y - (float)iy;
    const uint8_t *p = img + iy * W + ix;
    const float top = (float)p[0] + ax * ((float)p[1] - (float)p[0]);
    const float bot = (float)p[W] + ax * ((float)p[W + 1] - (float)p[W]);
    return top + ay * (bot - top);
}

static void rotvec_to_mat(const float r[3], float R[9])
{
    const float a = sqrtf(r[0] * r[0] + r[1] * r[1] + r[2] * r[2]);
    if (a < 1e-9f) {
        memset(R, 0, 9 * sizeof(float));
        R[0] = R[4] = R[8] = 1.0f;
        return;
    }
    const float x = r[0] / a, y = r[1] / a, z = r[2] / a;
    const float c = cosf(a), s = sinf(a), C = 1.0f - c;
    R[0] = c + x * x * C;     R[1] = x * y * C - z * s; R[2] = x * z * C + y * s;
    R[3] = y * x * C + z * s; R[4] = c + y * y * C;     R[5] = y * z * C - x * s;
    R[6] = z * x * C - y * s; R[7] = z * y * C + x * s; R[8] = c + z * z * C;
}

static void render_rotated_shifted(const vh_camera *cam, const uint8_t *key,
                                   const float R[9], float sx, float sy,
                                   uint8_t *cur)
{
    for (int v = 0; v < H; v++) {
        for (int u = 0; u < W; u++) {
            const float xu = ((float)u - sx - cam->cx) / cam->fx;
            const float yu = ((float)v - sy - cam->cy) / cam->fy;
            const float bx = R[0] * xu + R[1] * yu + R[2];
            const float by = R[3] * xu + R[4] * yu + R[5];
            const float bz = R[6] * xu + R[7] * yu + R[8];
            const float ku = cam->fx * (bx / bz) + cam->cx;
            const float kv = cam->fy * (by / bz) + cam->cy;
            cur[v * W + u] = (uint8_t)(sample_clamped(key, ku, kv) + 0.5f);
        }
    }
}

/* ---------- tests ---------- */

static const float R_CB_IDENTITY[9] = { 1, 0, 0, 0, 1, 0, 0, 0, 1 };

static void run_case(const vh_camera *cam, const char *name,
                     const float rotvec[3], float sx, float sy, bool feed_gyro)
{
    printf("%s:\n", name);

    static vh_ctx ctx;
    vh_params prm;
    memset(&prm, 0, sizeof(prm));
    prm.cam = *cam;
    memcpy(prm.r_cb, R_CB_IDENTITY, sizeof(prm.r_cb));
    prm.auto_rekey = false;
    vh_init(&ctx, &prm);

    const uint64_t t0 = 1000000, t1 = t0 + 100000;
    vh_image key_img = { g_base, W, H, W };
    const int n = vh_set_keyframe(&ctx, &key_img, t0);
    if (n <= 0) {
        CHECK(false, "keyframe accepted");
        return;
    }

    float R[9];
    rotvec_to_mat(rotvec, R);
    if (feed_gyro) {
        const float T = (float)(t1 - t0) * 1e-6f;
        const float w[3] = { rotvec[0] / T, rotvec[1] / T, rotvec[2] / T };
        for (uint64_t t = t0 + 1000; t <= t1; t += 1000) /* 1 kHz */
            vh_gyro(&ctx, t, w[0], w[1], w[2]);
    }

    render_rotated_shifted(cam, g_base, R, sx, sy, g_cur);
    vh_image cur_img = { g_cur, W, H, W };
    const vh_result res = vh_process_frame(&ctx, &cur_img, t1);

    printf("  status=%d tracked=%d/%d res=(%.2f, %.2f) div=%.2f px\n",
           (int)res.status, res.n_tracked, res.n_keyframe,
           (double)res.res_x_px, (double)res.res_y_px, (double)res.divergence_px);
    CHECK(res.status == VH_STATUS_OK || res.status == VH_STATUS_DEGRADED,
          "hold output valid");
    CHECK(fabsf(res.res_x_px - sx) < 0.35f,
          "x residual %.2f px matches injected %.2f px", (double)res.res_x_px, (double)sx);
    CHECK(fabsf(res.res_y_px - sy) < 0.35f,
          "y residual %.2f px matches injected %.2f px", (double)res.res_y_px, (double)sy);
}

/*
 * Simulated hover: N frames at 30 fps around one keyframe, no re-key.
 * Injected motion: slow sinusoidal drift (amplitude ax/ay px) plus
 * sinusoidal rotation jitter about all three axes fed through the gyro path
 * (peak rate ~23 deg/s — well above anything in the DCS forward-flight
 * archive). Checks per-frame recovery error and end-of-run drift.
 */
static void test_hover_drift(const vh_camera *cam)
{
    enum { N = 300 };                        /* 10 s at 30 fps */
    const float ax = 6.0f, ay = 4.0f;        /* drift amplitude, px */
    const float rot_amp[3] = { 0.05f, 0.04f, 0.06f }; /* rad, per-axis */

    printf("Real-texture hover, %d frames, rotation jitter + drift:\n", N);

    static vh_ctx ctx;
    vh_params prm;
    memset(&prm, 0, sizeof(prm));
    prm.cam = *cam;
    memcpy(prm.r_cb, R_CB_IDENTITY, sizeof(prm.r_cb));
    prm.auto_rekey = false;
    vh_init(&ctx, &prm);

    const uint64_t t0 = 1000000;
    const uint64_t dt = 33333;               /* 30 fps */
    vh_image key_img = { g_base, W, H, W };
    const int nkey = vh_set_keyframe(&ctx, &key_img, t0);
    printf("  keyframe features: %d\n", nkey);

    float worst = 0.0f, worst_rate = 0.0f;
    float prev_theta[3] = { 0, 0, 0 };
    /* Ground-truth rotation is the path-ordered product of the per-frame
     * increments — the same composition a body-mounted gyro sees. Composing
     * exp(theta(k)) directly instead would differ by the accumulated
     * commutator (BCH) terms: ~1 px after three loops at these amplitudes. */
    float R_true[9] = { 1, 0, 0, 0, 1, 0, 0, 0, 1 };
    int n_valid = 0, min_tracked = 9999;
    float final_err = -1.0f;

    for (int k = 1; k <= N; k++) {
        const float ph = 2.0f * 3.14159265f * (float)k / (float)N;
        /* two full drift cycles, nine rotation cycles; both return to zero
         * at k == N so the final frame must land back on the keyframe. */
        const float sx = ax * sinf(2.0f * ph);
        const float sy = ay * sinf(2.0f * ph + 0.7f) - ay * sinf(0.7f);
        float theta[3];
        for (int i = 0; i < 3; i++)
            theta[i] = rot_amp[i] * sinf(9.0f * ph + (float)i);
        /* constant-rate gyro over the frame interval; per-axis rates from
         * the theta delta (axes vary slowly, treat as independent) */
        const uint64_t ta = t0 + (uint64_t)(k - 1) * dt;
        const uint64_t tb = t0 + (uint64_t)k * dt;
        const float T = (float)(tb - ta) * 1e-6f;
        float w[3], dth[3], rate = 0.0f;
        for (int i = 0; i < 3; i++) {
            dth[i] = theta[i] - prev_theta[i];
            w[i] = dth[i] / T;
            rate += w[i] * w[i];
        }
        rate = sqrtf(rate) * 57.2958f;
        for (uint64_t t = ta + 1000; t <= tb; t += 1000)
            vh_gyro(&ctx, t, w[0], w[1], w[2]);
        memcpy(prev_theta, theta, sizeof(theta));

        /* R_true = R_true * exp(dth): body-frame increment, right-multiplied
         * exactly as the gyro integrator composes it. */
        float dR[9], Rn[9];
        rotvec_to_mat(dth, dR);
        for (int r = 0; r < 3; r++)
            for (int c = 0; c < 3; c++)
                Rn[r * 3 + c] = R_true[r * 3 + 0] * dR[0 * 3 + c] +
                                R_true[r * 3 + 1] * dR[1 * 3 + c] +
                                R_true[r * 3 + 2] * dR[2 * 3 + c];
        memcpy(R_true, Rn, sizeof(R_true));

        render_rotated_shifted(cam, g_base, R_true, sx, sy, g_cur);
        vh_image cur_img = { g_cur, W, H, W };
        const vh_result res = vh_process_frame(&ctx, &cur_img, tb);

        if (res.status != VH_STATUS_OK && res.status != VH_STATUS_DEGRADED) {
            CHECK(false, "frame %d lost track (status %d)", k, (int)res.status);
            return;
        }
        const float err = hypotf(res.res_x_px - sx, res.res_y_px - sy);
        if (err > worst) { worst = err; worst_rate = rate; }
        if (res.n_tracked < min_tracked) min_tracked = res.n_tracked;
        if (k == N) final_err = err;
        n_valid++;
    }

    printf("  %d/%d frames valid, min tracked %d, worst error %.3f px (at %.0f deg/s), final-frame error %.3f px\n",
           n_valid, N, min_tracked, (double)worst, (double)worst_rate, (double)final_err);
    CHECK(n_valid == N, "all %d frames produced valid output", N);
    CHECK(worst < 0.35f, "worst per-frame recovery error %.3f px < 0.35 px", (double)worst);
    CHECK(final_err < 0.15f,
          "no drift: final frame back at keyframe within %.3f px", (double)final_err);
}

/*
 * Gyro-bias scenario: the drone hovers (small translation sinusoid, ZERO
 * true rotation) but the gyro reports a constant bias. Without correction
 * the spurious integrated rotation drifts the prediction — and the residual
 * with it — for the life of the keyframe. vh_bias must observe that drift,
 * converge to the injected bias, and flatten the residual.
 */
static void test_bias_estimator(const vh_camera *cam, bool use_estimator)
{
    enum { N = 450 };                          /* 15 s at 30 fps */
    const float BIAS[3] = { 0.00873f, -0.00524f, 0.0f }; /* 0.5, -0.3 deg/s */

    printf("Hover with 0.5/-0.3 deg/s gyro bias, estimator %s:\n",
           use_estimator ? "ON" : "OFF");

    static vh_ctx ctx;
    vh_params prm;
    memset(&prm, 0, sizeof(prm));
    prm.cam = *cam;
    memcpy(prm.r_cb, R_CB_IDENTITY, sizeof(prm.r_cb));
    prm.auto_rekey = false;
    vh_init(&ctx, &prm);

    vh_bias est;
    vh_bias_init(&est, 3.0f); /* short tau for a 10 s test; use ~20 s in flight */
    est.max_res_px = 12.0f;

    const uint64_t t0 = 1000000, dt = 33333;
    vh_image key_img = { g_base, W, H, W };
    vh_set_keyframe(&ctx, &key_img, t0);

    float worst = 0.0f, final_err = 0.0f, err_3s_before_end = 0.0f;
    for (int k = 1; k <= N; k++) {
        const float ph = 2.0f * 3.14159265f * (float)k / (float)N;
        const float sx = 2.0f * sinf(3.0f * ph);
        const float sy = 1.5f * sinf(3.0f * ph + 0.9f) - 1.5f * sinf(0.9f);

        const uint64_t ta = t0 + (uint64_t)(k - 1) * dt;
        const uint64_t tb = t0 + (uint64_t)k * dt;
        for (uint64_t t = ta + 1000; t <= tb; t += 1000)
            vh_gyro(&ctx, t, BIAS[0], BIAS[1], BIAS[2]); /* pure lies */

        /* True motion: translation only, no rotation. */
        float I[9] = { 1, 0, 0, 0, 1, 0, 0, 0, 1 };
        render_rotated_shifted(cam, g_base, I, sx, sy, g_cur);
        vh_image cur_img = { g_cur, W, H, W };
        const vh_result res = vh_process_frame(&ctx, &cur_img, tb);
        if (res.status != VH_STATUS_OK && res.status != VH_STATUS_DEGRADED) {
            CHECK(false, "frame %d lost track (residual drifted out of range)", k);
            return;
        }
        if (use_estimator && vh_bias_update(&est, &res, cam))
            vh_rot_set_bias(&ctx.rot, est.bias_c);

        const float err = hypotf(res.res_x_px - sx, res.res_y_px - sy);
        if (err > worst) worst = err;
        if (k == N - 90) err_3s_before_end = err;
        if (k == N) final_err = err;
    }

    printf("  worst error %.2f px, final error %.2f px", (double)worst, (double)final_err);
    if (use_estimator)
        printf(", bias est (%.3f, %.3f) deg/s vs injected (0.500, -0.300)",
               (double)(est.bias_c[0] * 57.2958f), (double)(est.bias_c[1] * 57.2958f));
    printf("\n");

    if (use_estimator) {
        /* The angle error accumulated DURING convergence stays in q until
         * the next re-key — a constant offset, not drift. The claim to test
         * is that the drift has stopped: uncorrected it moves ~3.9 px in
         * the same 3 s window. */
        const float late_drift = fabsf(final_err - err_3s_before_end);
        CHECK(late_drift < 0.75f,
              "drift stopped: %.2f px over final 3 s (uncorrected: ~3.9 px)",
              (double)late_drift);
        CHECK(fabsf(est.bias_c[0] * 57.2958f - 0.5f) < 0.2f &&
              fabsf(est.bias_c[1] * 57.2958f + 0.3f) < 0.2f,
              "bias recovered within 0.2 deg/s");
    } else {
        CHECK(worst > 5.0f, "uncorrected bias drifts the residual (worst %.2f px)",
              (double)worst);
    }
}

int main(int argc, char **argv)
{
    const char *path = argc > 1 ? argv[1]
        : "../testdata/dcs/processed/easyair_001_forpost_snow_sun_200m_level_straight.vhr";
    const uint32_t frame = argc > 2 ? (uint32_t)atoi(argv[2]) : 0;

    const vh_camera cam = load_vhr_frame(path, frame);
    printf("Real frame %u from %s (fx=%.1f fy=%.1f)\n\n", frame, path,
           (double)cam.fx, (double)cam.fy);

    {
        printf("FAST-9 on real frame:\n");
        vh_image img = { g_base, W, H, W };
        vh_corner corners[VH_MAX_FEATURES];
        const int n = vh_fast_detect(&img, corners);
        printf("  detected %d corners (max %d)\n", n, VH_MAX_FEATURES);
        CHECK(n >= VH_MIN_TRACKED * 2, "enough corners on real texture (got %d)", n);
    }

    run_case(&cam, "Zero motion (real frame)", (float[3]){ 0, 0, 0 }, 0.0f, 0.0f, false);
    run_case(&cam, "Sub-pixel translation (+2.3, -1.6) px", (float[3]){ 0, 0, 0 },
             2.3f, -1.6f, false);
    /* ~2.1 deg combined: ~5 px of image motion at fx~150 to cancel */
    run_case(&cam, "Pure rotation, gyro-compensated",
             (float[3]){ 0.020f, -0.014f, 0.025f }, 0.0f, 0.0f, true);
    run_case(&cam, "Rotation + translation (+2.0, +1.5) px",
             (float[3]){ -0.016f, 0.018f, -0.020f }, 2.0f, 1.5f, true);

    test_hover_drift(&cam);
    test_bias_estimator(&cam, false);
    test_bias_estimator(&cam, true);

    printf("\n%s (%d failure%s)\n", g_failures ? "FAILED" : "ALL TESTS PASSED",
           g_failures, g_failures == 1 ? "" : "s");
    return g_failures ? 1 : 0;
}
