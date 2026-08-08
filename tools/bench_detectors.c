/*
 * bench_detectors.c — compare corner detectors as vio_hold front-ends on
 * real archive frames: latency, scene coverage, and end-to-end hold quality
 * (the detector's corners seeded into the actual pipeline, then a warp-hover
 * with exact ground truth).
 *
 * Detectors (all: 12x8 grid bucketing, best per cell, 12 px margin, <=96):
 *   fast20  — FAST-9, threshold 20, min score 60 (the library's front-end)
 *   fast10  — FAST-9, threshold 10, min score 30 (adaptive-retry candidate)
 *   shitom  — Shi-Tomasi min-eigenvalue, Sobel 3x3, 5x5 window, 1% floor
 *   harris  — Harris R = det - 0.04 tr^2, same sums, 1% floor
 *
 * Usage: bench_detectors <dir-with-vhr-files>
 */
#define _POSIX_C_SOURCE 199309L
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "vio_hold/vh_hold.h"

#define W VH_IMG_W
#define H VH_IMG_H
#define MAXC VH_MAX_FEATURES

typedef struct { float x, y; } pt;

/* ---------- .vhr frame loader ---------- */
static uint32_t rd_u32(FILE *f) { uint8_t b[4] = {0}; if (fread(b,1,4,f)!=4){} return (uint32_t)b[0]|(uint32_t)b[1]<<8|(uint32_t)b[2]<<16|(uint32_t)b[3]<<24; }
static int load_frame(const char *path, uint32_t idx, uint8_t *dst)
{
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "missing: %s\n", path); return 0; }
    fseek(f, 32, SEEK_SET);
    uint32_t ng = rd_u32(f), nf = rd_u32(f);
    if (idx >= nf) { fclose(f); return 0; }
    fseek(f, 40L + ng*20L + (long)idx*(8L + W*H) + 8, SEEK_SET);
    int ok = fread(dst, 1, W*H, f) == W*H;
    fclose(f);
    return ok;
}

/* ---------- detectors ---------- */
static const int8_t CDX[16] = { 0,1,2,3,3,3,2,1,0,-1,-2,-3,-3,-3,-2,-1 };
static const int8_t CDY[16] = { -3,-3,-2,-1,0,1,2,3,3,3,2,1,0,-1,-2,-3 };
static inline int has_arc9(uint32_t m16)
{
    uint32_t m = m16 | (m16 << 16);
    m &= m << 1; m &= m << 2; m &= m << 4; m &= m << 1;
    return m != 0;
}
static int det_fast(const uint8_t *px, int t, int minsc, pt *out)
{
    int32_t off[16];
    for (int i = 0; i < 16; i++) off[i] = (int32_t)CDY[i]*W + CDX[i];
    float bx[MAXC], by[MAXC]; int32_t bs[MAXC];
    for (int i = 0; i < MAXC; i++) bs[i] = -1;
    for (int y = VH_DET_MARGIN; y < H-VH_DET_MARGIN; y++) {
        const uint8_t *row = px + (int32_t)y*W;
        const int cr = (y*VH_GRID_ROWS)/H;
        for (int x = VH_DET_MARGIN; x < W-VH_DET_MARGIN; x++) {
            const uint8_t *p = row + x;
            const int c = *p, hi = c+t, lo = c-t;
            int nb = 0, nd = 0;
            for (int i = 0; i < 16; i += 4) {
                const int v = p[off[i]];
                nb += (v > hi); nd += (v < lo);
            }
            if (nb < 2 && nd < 2) continue;
            uint32_t mb = 0, md = 0; int32_t sb = 0, sd = 0;
            for (int i = 0; i < 16; i++) {
                const int v = p[off[i]];
                if (v > hi) { mb |= 1u<<i; sb += v-c; }
                else if (v < lo) { md |= 1u<<i; sd += c-v; }
            }
            int32_t sc;
            if (has_arc9(mb)) sc = sb;
            else if (has_arc9(md)) sc = sd;
            else continue;
            if (sc < minsc) continue;
            const int cell = cr*VH_GRID_COLS + (x*VH_GRID_COLS)/W;
            if (sc > bs[cell]) { bs[cell] = sc; bx[cell] = (float)x; by[cell] = (float)y; }
        }
    }
    int n = 0;
    for (int i = 0; i < MAXC; i++) if (bs[i] >= 0) { out[n].x = bx[i]; out[n].y = by[i]; n++; }
    return n;
}

/* Shi-Tomasi / Harris via Sobel + 5x5 window sums (response map + 1% floor).
 * Host implementation uses full-frame scratch; an MCU port would run a
 * sliding window — arithmetic per pixel is what the timing measures. */
static int16_t g_gx[W*H], g_gy[W*H];
static float g_resp[W*H];
static int det_eig(const uint8_t *px, int harris, pt *out)
{
    for (int y = 1; y < H-1; y++)
        for (int x = 1; x < W-1; x++) {
            const uint8_t *p = px + y*W + x;
            g_gx[y*W+x] = (int16_t)((p[-W+1]+2*p[1]+p[W+1]) - (p[-W-1]+2*p[-1]+p[W-1]));
            g_gy[y*W+x] = (int16_t)((p[W-1]+2*p[W]+p[W+1]) - (p[-W-1]+2*p[-W]+p[-W+1]));
        }
    float rmax = 0.f;
    const int R = 2; /* 5x5 window */
    for (int y = VH_DET_MARGIN; y < H-VH_DET_MARGIN; y++)
        for (int x = VH_DET_MARGIN; x < W-VH_DET_MARGIN; x++) {
            int64_t sxx = 0, syy = 0, sxy = 0;
            for (int j = -R; j <= R; j++)
                for (int i = -R; i <= R; i++) {
                    const int32_t gx = g_gx[(y+j)*W + x+i], gy = g_gy[(y+j)*W + x+i];
                    sxx += gx*gx; syy += gy*gy; sxy += gx*gy;
                }
            const float a = (float)sxx, b = (float)sxy, c = (float)syy;
            float r;
            if (harris) r = (a*c - b*b) - 0.04f*(a+c)*(a+c);
            else        r = 0.5f*((a+c) - sqrtf((a-c)*(a-c) + 4.f*b*b));
            g_resp[y*W+x] = r;
            if (r > rmax) rmax = r;
        }
    const float floor_ = 0.01f * rmax;
    float bx[MAXC], by[MAXC], bs[MAXC];
    for (int i = 0; i < MAXC; i++) bs[i] = -1.f;
    if (rmax <= 0.f) return 0;
    for (int y = VH_DET_MARGIN; y < H-VH_DET_MARGIN; y++) {
        const int cr = (y*VH_GRID_ROWS)/H;
        for (int x = VH_DET_MARGIN; x < W-VH_DET_MARGIN; x++) {
            const float r = g_resp[y*W+x];
            if (r < floor_) continue;
            const int cell = cr*VH_GRID_COLS + (x*VH_GRID_COLS)/W;
            if (r > bs[cell]) { bs[cell] = r; bx[cell] = (float)x; by[cell] = (float)y; }
        }
    }
    int n = 0;
    for (int i = 0; i < MAXC; i++) if (bs[i] >= 0.f) { out[n].x = bx[i]; out[n].y = by[i]; n++; }
    return n;
}

typedef int (*det_fn)(const uint8_t *, pt *);
static int d_fast20(const uint8_t *p, pt *o) { return det_fast(p, 20, 60, o); }
static int d_fast10(const uint8_t *p, pt *o) { return det_fast(p, 10, 30, o); }
static int d_shitom(const uint8_t *p, pt *o) { return det_eig(p, 0, o); }
static int d_harris(const uint8_t *p, pt *o) { return det_eig(p, 1, o); }

/* ---------- warp-hover quality harness (as in test_real_texture) -------- */
static uint8_t g_base[W*H], g_cur[W*H];
static float sample_clamped(const uint8_t *img, float x, float y)
{
    if (x < 0) x = 0;
    if (y < 0) y = 0;
    if (x > W-2) x = W-2;
    if (y > H-2) y = H-2;
    const int ix = (int)x, iy = (int)y;
    const float ax = x-ix, ay = y-iy;
    const uint8_t *p = img + iy*W + ix;
    const float top = p[0] + ax*(p[1]-p[0]);
    const float bot = p[W] + ax*(p[W+1]-p[W]);
    return top + ay*(bot-top);
}
static void rotvec_to_mat(const float r[3], float R[9])
{
    const float a = sqrtf(r[0]*r[0]+r[1]*r[1]+r[2]*r[2]);
    if (a < 1e-9f) { memset(R,0,36); R[0]=R[4]=R[8]=1.f; return; }
    const float x=r[0]/a, y=r[1]/a, z=r[2]/a, c=cosf(a), s=sinf(a), C=1.f-c;
    R[0]=c+x*x*C; R[1]=x*y*C-z*s; R[2]=x*z*C+y*s;
    R[3]=y*x*C+z*s; R[4]=c+y*y*C; R[5]=y*z*C-x*s;
    R[6]=z*x*C-y*s; R[7]=z*y*C+x*s; R[8]=c+z*z*C;
}
static void render(const vh_camera *cam, const float R[9], float sx, float sy)
{
    for (int v = 0; v < H; v++)
        for (int u = 0; u < W; u++) {
            const float xu = ((float)u-sx-cam->cx)/cam->fx;
            const float yu = ((float)v-sy-cam->cy)/cam->fy;
            const float bx = R[0]*xu+R[1]*yu+R[2];
            const float by = R[3]*xu+R[4]*yu+R[5];
            const float bz = R[6]*xu+R[7]*yu+R[8];
            g_cur[v*W+u] = (uint8_t)(sample_clamped(g_base,
                cam->fx*(bx/bz)+cam->cx, cam->fy*(by/bz)+cam->cy) + 0.5f);
        }
}

/* seed ctx keyframe from an arbitrary corner list (mirrors vh_set_keyframe) */
static vh_ctx g_ctx;
static int seed_keyframe(const vh_camera *cam, const pt *c, int n, uint64_t t)
{
    vh_params prm; memset(&prm, 0, sizeof prm);
    prm.cam = *cam;
    prm.r_cb[0] = prm.r_cb[4] = prm.r_cb[8] = 1.f;
    vh_init(&g_ctx, &prm);
    if (n < VH_MIN_TRACKED) return 0;
    memcpy(g_ctx.key_img, g_base, W*H);
    vh_image key = { g_ctx.key_img, W, H, W };
    vh_pyr_build(&g_ctx.key_pyr, &key, g_ctx.key_store);
    for (int i = 0; i < n; i++) {
        g_ctx.fx[i] = c[i].x; g_ctx.fy[i] = c[i].y;
        g_ctx.bearing[i] = vh_cam_pixel_to_bearing(cam, c[i].x, c[i].y);
        g_ctx.active[i] = true;
    }
    g_ctx.n_key = n; g_ctx.key_t_us = t; g_ctx.have_key = true;
    vh_rot_rekey(&g_ctx.rot, t);
    return n;
}

typedef struct { int n_key, min_trk, valid; float worst; } hover_res;
static hover_res hover_quality(const vh_camera *cam, det_fn fn)
{
    hover_res hr = { 0, 9999, 0, -1.f };
    pt c[MAXC];
    const int n = fn(g_base, c);
    hr.n_key = n;
    const uint64_t t0 = 1000000, dt = 33333;
    if (seed_keyframe(cam, c, n, t0) < VH_MIN_TRACKED) return hr;
    const int N = 150;
    float prev_th[3] = { 0, 0, 0 };
    float Rt[9] = { 1,0,0, 0,1,0, 0,0,1 };
    hr.worst = 0.f;
    for (int k = 1; k <= N; k++) {
        const float ph = 6.2831853f*k/N;
        const float sx = 4.f*sinf(2.f*ph), sy = 3.f*sinf(2.f*ph+0.7f) - 3.f*sinf(0.7f);
        float th[3], dth[3];
        for (int i = 0; i < 3; i++) {
            th[i] = 0.025f*sinf(3.f*ph + (float)i);
            dth[i] = th[i] - prev_th[i];
        }
        memcpy(prev_th, th, sizeof th);
        const uint64_t ta = t0 + (uint64_t)(k-1)*dt, tb = t0 + (uint64_t)k*dt;
        const float T = (float)(tb-ta)*1e-6f;
        for (uint64_t t = ta+1000; t <= tb; t += 1000)
            vh_gyro(&g_ctx, t, dth[0]/T, dth[1]/T, dth[2]/T);
        float dR[9], Rn[9];
        rotvec_to_mat(dth, dR);
        for (int r = 0; r < 3; r++)
            for (int cc2 = 0; cc2 < 3; cc2++)
                Rn[r*3+cc2] = Rt[r*3]*dR[cc2] + Rt[r*3+1]*dR[3+cc2] + Rt[r*3+2]*dR[6+cc2];
        memcpy(Rt, Rn, sizeof Rt);
        render(cam, Rt, sx, sy);
        vh_image img = { g_cur, W, H, W };
        vh_result r = vh_process_frame(&g_ctx, &img, tb);
        if (r.status != VH_STATUS_OK && r.status != VH_STATUS_DEGRADED) continue;
        hr.valid++;
        if (r.n_tracked < hr.min_trk) hr.min_trk = r.n_tracked;
        const float err = hypotf(r.res_x_px - sx, r.res_y_px - sy);
        if (err > hr.worst) hr.worst = err;
    }
    return hr;
}

static double now_us(void)
{
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec*1e6 + ts.tv_nsec*1e-3;
}

int main(int argc, char **argv)
{
    const char *dir = argc > 1 ? argv[1] : "../testdata/dcs";
    char path[512];
    struct { const char *name, *file; uint32_t idx; } scenes[] = {
        { "snow",     "processed/easyair_001_forpost_snow_sun_200m_level_straight.vhr", 0 },
        { "terrain",  "processed/orbitreq_01_mq9_500m_snake.vhr", 0 },
        { "urban_fog","sweep/easyair_017_ka27_urban_fog_150m_level_straight.vhr", 1000 },
        { "whiteout", "sweep/easyair_014_rq1_snow_precip_150m_level_straight.vhr", 80 },
        { "clouds",   "sweep/easyair_007_mi8_sea_cloud_225m_level_straight.vhr", 300 },
        { "sea",      "sweep/easyair_003_uh1_sea_sun_175m_level_straight.vhr", 200 },
    };
    struct { const char *name; det_fn fn; } dets[] = {
        { "fast20", d_fast20 }, { "fast10", d_fast10 },
        { "shitom", d_shitom }, { "harris", d_harris },
    };
    static uint8_t frames[6][W*H];
    for (int s = 0; s < 6; s++) {
        snprintf(path, sizeof path, "%s/%s", dir, scenes[s].file);
        if (!load_frame(path, scenes[s].idx, frames[s])) return 1;
    }

    printf("== corner counts per scene (max %d)\n%-8s", MAXC, "");
    for (int s = 0; s < 6; s++) printf("%10s", scenes[s].name);
    printf("\n");
    pt c[MAXC];
    for (int d = 0; d < 4; d++) {
        printf("%-8s", dets[d].name);
        for (int s = 0; s < 6; s++) printf("%10d", dets[d].fn(frames[s], c));
        printf("\n");
    }

    printf("\n== latency, us/frame at %dx%d (host; mean of 50 runs x 6 scenes)\n", W, H);
    for (int d = 0; d < 4; d++) {
        const double a = now_us();
        for (int rep = 0; rep < 50; rep++)
            for (int s = 0; s < 6; s++) dets[d].fn(frames[s], c);
        const double b = now_us();
        printf("%-8s %8.0f us\n", dets[d].name, (b-a)/300.0);
    }

    vh_camera cam = { 149.4f, 148.2f, 160.f, 120.f, 0, 0, 0, 0 };
    printf("\n== end-to-end warp-hover quality (150 frames, drift + rotation jitter)\n");
    printf("%-8s %-10s %7s %7s %7s %10s\n", "", "scene", "n_key", "valid", "minTrk", "worst_px");
    for (int s = 0; s < 6; s++) {
        if (s != 0 && s != 3) continue; /* snow (easy) and whiteout (hard) */
        memcpy(g_base, frames[s], W*H);
        for (int d = 0; d < 4; d++) {
            hover_res hr = hover_quality(&cam, dets[d].fn);
            if (hr.worst < 0.f)
                printf("%-8s %-10s %7d %7s %7s %10s\n", dets[d].name, scenes[s].name,
                       hr.n_key, "-", "-", "REJECTED");
            else
                printf("%-8s %-10s %7d %5d/150 %7d %10.3f\n", dets[d].name, scenes[s].name,
                       hr.n_key, hr.valid, hr.min_trk, (double)hr.worst);
        }
    }
    return 0;
}
