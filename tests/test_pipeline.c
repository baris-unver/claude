/*
 * Host-side numerical tests for the vio_hold pipeline.
 *
 * Synthesizes a textured scene, applies known image motions (pure shift,
 * pure rotation via the camera model, rotation + shift) and checks that the
 * pipeline recovers the translation residual and cancels rotation.
 *
 * Build & run:  make test
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "vio_hold/vh_hold.h"
#include "vio_hold/vh_fast.h"

#define W VH_IMG_W
#define H VH_IMG_H

static uint8_t g_key[W * H];
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

/* ---------- synthetic scene ---------- */

static uint32_t xorshift32(uint32_t *s)
{
    uint32_t x = *s;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    return *s = x;
}

/* Random texture with a mild 3x3 blur so it has trackable gradients. */
static void make_texture(uint8_t *img)
{
    static uint8_t raw[W * H];
    uint32_t seed = 0xC0FFEEu;
    for (int i = 0; i < W * H; i++)
        raw[i] = (uint8_t)(xorshift32(&seed) & 0xFF);

    for (int y = 0; y < H; y++) {
        for (int x = 0; x < W; x++) {
            int sum = 0, n = 0;
            for (int j = -1; j <= 1; j++) {
                for (int i = -1; i <= 1; i++) {
                    const int yy = y + j, xx = x + i;
                    if (yy >= 0 && yy < H && xx >= 0 && xx < W) {
                        sum += raw[yy * W + xx];
                        n++;
                    }
                }
            }
            img[y * W + x] = (uint8_t)(sum / n);
        }
    }
}

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

/* ---------- rotation utilities (test-side ground truth) ---------- */

/* Rodrigues: rotation vector -> row-major 3x3. */
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

/*
 * Render the view after camera rotation R_delta (key->current) plus an image
 * shift (sx, sy). A world direction with key-frame bearing b appears in the
 * current frame at pixel  proj(R_delta^T b) + s,  so the inverse warp is
 * cur(u) = key(proj(R_delta * bearing(u - s))).
 */
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

static vh_camera test_cam(void)
{
    vh_camera c = { 250.0f, 250.0f, (float)W / 2.0f, (float)H / 2.0f,
                    0.0f, 0.0f, 0.0f, 0.0f };
    return c;
}

static void test_fast_detect(void)
{
    printf("FAST-9 detection:\n");
    vh_image img = { g_key, W, H, W };
    vh_corner corners[VH_MAX_FEATURES];
    const int n = vh_fast_detect(&img, corners);
    printf("  detected %d corners (max %d)\n", n, VH_MAX_FEATURES);
    CHECK(n >= 40, "at least 40 corners on synthetic texture (got %d)", n);
}

static void test_camera_roundtrip(void)
{
    printf("Camera model distort/undistort round trip:\n");
    vh_camera c = test_cam();
    c.k1 = -0.28f;
    c.k2 = 0.07f;
    c.p1 = 0.0005f;
    c.p2 = -0.0003f;

    float worst = 0.0f;
    for (int v = 20; v < H - 20; v += 40) {
        for (int u = 20; u < W - 20; u += 40) {
            const vh_vec3 b = vh_cam_pixel_to_bearing(&c, (float)u, (float)v);
            float ru, rv;
            if (!vh_cam_bearing_to_pixel(&c, b, &ru, &rv)) {
                worst = 1e9f;
                continue;
            }
            const float e = hypotf(ru - (float)u, rv - (float)v);
            if (e > worst)
                worst = e;
        }
    }
    CHECK(worst < 0.02f, "worst round-trip error %.4f px < 0.02 px", (double)worst);
}

static void run_case(const char *name, const float rotvec[3], float sx, float sy,
                     bool feed_gyro)
{
    printf("%s:\n", name);
    const vh_camera cam = test_cam();

    static vh_ctx ctx; /* too big for the stack */
    vh_params prm;
    memset(&prm, 0, sizeof(prm));
    prm.cam = cam;
    memcpy(prm.r_cb, R_CB_IDENTITY, sizeof(prm.r_cb));
    prm.auto_rekey = false;
    vh_init(&ctx, &prm);

    const uint64_t t0 = 1000000; /* 1 s */
    const uint64_t t1 = t0 + 100000; /* +100 ms */

    vh_image key_img = { g_key, W, H, W };
    const int n = vh_set_keyframe(&ctx, &key_img, t0);
    if (n <= 0) {
        CHECK(false, "keyframe accepted");
        return;
    }

    /* Ground-truth rotation over [t0, t1] and matching gyro stream. */
    float R[9];
    rotvec_to_mat(rotvec, R);
    if (feed_gyro) {
        const float T = (float)(t1 - t0) * 1e-6f;
        const float w[3] = { rotvec[0] / T, rotvec[1] / T, rotvec[2] / T };
        for (uint64_t t = t0 + 500; t <= t1; t += 500) /* 2 kHz */
            vh_gyro(&ctx, t, w[0], w[1], w[2]);
    }

    render_rotated_shifted(&cam, g_key, R, sx, sy, g_cur);

    vh_image cur_img = { g_cur, W, H, W };
    const vh_result res = vh_process_frame(&ctx, &cur_img, t1);

    printf("  status=%d tracked=%d/%d res=(%.2f, %.2f) div=%.2f px\n",
           (int)res.status, res.n_tracked, res.n_keyframe,
           (double)res.res_x_px, (double)res.res_y_px, (double)res.divergence_px);

    CHECK(res.status == VH_STATUS_OK || res.status == VH_STATUS_DEGRADED,
          "hold output valid");
    CHECK(fabsf(res.res_x_px - sx) < 0.35f,
          "x residual %.2f px matches injected shift %.2f px", (double)res.res_x_px,
          (double)sx);
    CHECK(fabsf(res.res_y_px - sy) < 0.35f,
          "y residual %.2f px matches injected shift %.2f px", (double)res.res_y_px,
          (double)sy);
}

static void test_zero_motion(void)
{
    printf("Zero motion:\n");
    const vh_camera cam = test_cam();

    static vh_ctx ctx;
    vh_params prm;
    memset(&prm, 0, sizeof(prm));
    prm.cam = cam;
    memcpy(prm.r_cb, R_CB_IDENTITY, sizeof(prm.r_cb));
    vh_init(&ctx, &prm);

    vh_image key_img = { g_key, W, H, W };
    vh_set_keyframe(&ctx, &key_img, 1000000);
    const vh_result res = vh_process_frame(&ctx, &key_img, 1033333);
    printf("  tracked=%d res=(%.3f, %.3f)\n", res.n_tracked,
           (double)res.res_x_px, (double)res.res_y_px);
    CHECK(res.status == VH_STATUS_OK, "status OK");
    CHECK(fabsf(res.res_x_px) < 0.05f && fabsf(res.res_y_px) < 0.05f,
          "residual ~0 on identical frame");
}

int main(void)
{
    make_texture(g_key);

    test_fast_detect();
    test_camera_roundtrip();
    test_zero_motion();

    run_case("Pure translation (+3.7, -2.2) px", (float[3]){ 0, 0, 0 },
             3.7f, -2.2f, false);

    /* ~1.4 deg combined rotation: ~6 px of raw image motion that the gyro
     * compensation must cancel. */
    run_case("Pure rotation, gyro-compensated",
             (float[3]){ 0.015f, -0.010f, 0.020f }, 0.0f, 0.0f, true);

    run_case("Rotation + translation (+2.5, +1.8) px",
             (float[3]){ -0.012f, 0.014f, -0.015f }, 2.5f, 1.8f, true);

    /* Same rotation WITHOUT gyro feed: residual should now show the ~6 px of
     * uncompensated rotation-induced motion, proving compensation matters. */
    {
        printf("Rotation without gyro (sanity: compensation is doing work):\n");
        const vh_camera cam = test_cam();
        static vh_ctx ctx;
        vh_params prm;
        memset(&prm, 0, sizeof(prm));
        prm.cam = cam;
        memcpy(prm.r_cb, R_CB_IDENTITY, sizeof(prm.r_cb));
        vh_init(&ctx, &prm);
        vh_image key_img = { g_key, W, H, W };
        vh_set_keyframe(&ctx, &key_img, 1000000);
        float R[9];
        rotvec_to_mat((float[3]){ 0.015f, -0.010f, 0.020f }, R);
        render_rotated_shifted(&cam, g_key, R, 0.0f, 0.0f, g_cur);
        vh_image cur_img = { g_cur, W, H, W };
        const vh_result res = vh_process_frame(&ctx, &cur_img, 1100000);
        const float mag = hypotf(res.res_x_px, res.res_y_px);
        printf("  uncompensated residual magnitude: %.2f px\n", (double)mag);
        CHECK(mag > 2.0f, "uncompensated rotation leaks into residual (%.2f px)",
              (double)mag);
    }

    printf("\n%s (%d failure%s)\n", g_failures ? "FAILED" : "ALL TESTS PASSED",
           g_failures, g_failures == 1 ? "" : "s");
    return g_failures ? 1 : 0;
}
