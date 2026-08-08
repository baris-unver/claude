/*
 * profile_pipeline.c — per-stage host timing of the vio_hold pipeline on a
 * real frame sequence, as the basis for an MCU throughput estimate.
 *
 * Usage: profile_pipeline <file.vhr> [host_ghz]
 *
 * Reports per-frame wall time for: pyramid build, full vh_process_frame
 * (pyramid + prediction + KLT + median), keyframe capture, and FAST detect,
 * plus derived per-feature KLT cost. If host_ghz is given (or read from
 * /proc), also prints host cycles/frame and a Cortex-M55 estimate band
 * using a documented cycle-inflation factor: the M55 is a dual-issue
 * in-order core, so identical plain-C work typically costs 2-4x the cycles
 * of a wide out-of-order desktop core. This is a bounded estimate, not a
 * measurement — on-target profiling supersedes it.
 */
#define _POSIX_C_SOURCE 199309L
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "vio_hold/vh_hold.h"
#include "vio_hold/vh_fast.h"

#define W VH_IMG_W
#define H VH_IMG_H

static uint32_t rd_u32(FILE *f) { uint8_t b[4] = {0}; if (fread(b,1,4,f)!=4){} return (uint32_t)b[0]|(uint32_t)b[1]<<8|(uint32_t)b[2]<<16|(uint32_t)b[3]<<24; }
static double now_us(void)
{
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec*1e6 + (double)ts.tv_nsec*1e-3;
}

static vh_ctx g_ctx;
static uint8_t g_frames[400][W*H]; /* up to 400 frames resident */
static uint64_t g_t[400];

int main(int argc, char **argv)
{
    if (argc < 2) { fprintf(stderr, "usage: %s <file.vhr> [host_ghz]\n", argv[0]); return 2; }
    FILE *f = fopen(argv[1], "rb");
    if (!f) { perror(argv[1]); return 1; }
    fseek(f, 12, SEEK_SET);
    float fx = 0, fy = 0, cx = 0, cy = 0;
    if (fread(&fx, 4, 1, f) + fread(&fy, 4, 1, f) +
        fread(&cx, 4, 1, f) + fread(&cy, 4, 1, f) != 4) { fclose(f); return 1; }
    fseek(f, 32, SEEK_SET);
    uint32_t ng = rd_u32(f), nf = rd_u32(f);
    if (nf > 400) nf = 400;
    fseek(f, 40L + ng*20L, SEEK_SET);
    for (uint32_t i = 0; i < nf; i++) {
        uint8_t tb[8];
        if (fread(tb, 1, 8, f) != 8) { nf = i; break; }
        memcpy(&g_t[i], tb, 8);
        if (fread(g_frames[i], 1, W*H, f) != W*H) { nf = i; break; }
    }
    fclose(f);
    printf("%u frames loaded, fx=%.1f, sizeof(vh_ctx)=%zu bytes\n\n", nf, (double)fx, sizeof(vh_ctx));

    vh_params prm; memset(&prm, 0, sizeof prm);
    prm.cam.fx = fx; prm.cam.fy = fy; prm.cam.cx = cx; prm.cam.cy = cy;
    prm.r_cb[0] = prm.r_cb[4] = prm.r_cb[8] = 1.f;
    prm.auto_rekey = true;
    vh_init(&g_ctx, &prm);

    vh_image img0 = { g_frames[0], W, H, W };

    /* stage: FAST detect */
    vh_corner c[VH_MAX_FEATURES];
    double t0 = now_us();
    int reps = 200, n_corners = 0;
    for (int r = 0; r < reps; r++) n_corners = vh_fast_detect(&img0, c);
    (void)n_corners;
    double t_fast = (now_us() - t0) / reps;

    /* stage: pyramid build (into the ctx scratch) */
    t0 = now_us();
    for (int r = 0; r < reps; r++) vh_pyr_build(&g_ctx.cur_pyr, &img0, g_ctx.cur_store);
    double t_pyr = (now_us() - t0) / reps;

    /* stage: keyframe capture (detect + copy + pyramid + bearings) */
    t0 = now_us();
    for (int r = 0; r < reps; r++) vh_set_keyframe(&g_ctx, &img0, g_t[0]);
    double t_key = (now_us() - t0) / reps;

    /* stage: full per-frame processing over the real sequence */
    vh_set_keyframe(&g_ctx, &img0, g_t[0]);
    long total_tracked = 0; int calls = 0;
    t0 = now_us();
    for (uint32_t i = 1; i < nf; i++) {
        vh_image im = { g_frames[i], W, H, W };
        vh_result r = vh_process_frame(&g_ctx, &im, g_t[i]);
        total_tracked += r.n_tracked; calls++;
    }
    double t_frame = (now_us() - t0) / calls;
    double avg_trk = (double)total_tracked / calls;
    double t_klt_feat = (t_frame - t_pyr) / (avg_trk > 1 ? avg_trk : 1);

    printf("stage timings (host):\n");
    printf("  FAST-9 detect          %8.0f us\n", t_fast);
    printf("  pyramid build (3 lvl)  %8.0f us\n", t_pyr);
    printf("  keyframe capture       %8.0f us\n", t_key);
    printf("  process_frame (full)   %8.0f us   (%.1f features tracked avg)\n", t_frame, avg_trk);
    printf("  KLT per feature        %8.1f us\n", t_klt_feat);

    double ghz = argc > 2 ? atof(argv[2]) : 0.0;
    if (ghz <= 0) {
        FILE *cf = fopen("/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq", "r");
        if (cf) { long khz = 0; if (fscanf(cf, "%ld", &khz) == 1) ghz = (double)khz / 1e6; fclose(cf); }
    }
    if (ghz > 0) {
        const double cyc_frame = t_frame * ghz * 1e3;        /* host cycles */
        const double m55 = 800e6;                            /* 800 MHz */
        printf("\nhost clock %.2f GHz -> %.2f Mcycles/frame (host)\n", ghz, cyc_frame / 1e6);
        printf("Cortex-M55 @800 MHz estimate (plain C, x2..x4 cycle inflation):\n");
        for (double k = 2; k <= 4; k += 1)
            printf("  x%.0f: %6.1f ms/frame -> %5.1f Hz\n",
                   k, cyc_frame * k / m55 * 1e3, m55 / (cyc_frame * k));
        printf("(Helium/CMSIS-DSP vectorization of KLT patch ops and the FAST\n"
               " circle test recovers a large share of the inflation; memory\n"
               " placement per README: frames+ctx in AXISRAM.)\n");
    }
    return 0;
}
