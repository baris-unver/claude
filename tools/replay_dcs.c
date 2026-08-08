/*
 * replay_dcs.c — replay a preprocessed DCS sequence (.vhr, see
 * tools/dcs_extract.py) through the vio_hold pipeline.
 *
 * Usage: replay_dcs <file.vhr> [options]
 *   --tilt <deg>   override camera tilt about body right axis (+up)
 *   --no-gyro      do not feed gyro samples (rotation leaks into residual)
 *   --csv <path>   write per-frame results as CSV
 *   --dump <path>  write per-feature JSONL for visualization (requires a
 *                  build with -DVH_DEBUG_TRACKS, see Makefile replay_dump)
 *
 * The first frame becomes the keyframe; auto re-key is enabled, so on track
 * loss the current frame is promoted (as the flight controller would do).
 * Prints a per-frame line and a summary. Residuals are in pixels at
 * VH_IMG_W x VH_IMG_H resolution.
 */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "vio_hold/vh_hold.h"

#define MAGIC 0x56484452u

typedef struct {
    uint64_t t_us;
    float wx, wy, wz;
} gyro_rec;

static uint16_t rd_u16(FILE *f) { uint8_t b[2] = {0}; if (fread(b, 1, 2, f) != 2) {} return (uint16_t)(b[0] | b[1] << 8); }
static uint32_t rd_u32(FILE *f) { uint8_t b[4] = {0}; if (fread(b, 1, 4, f) != 4) {} return (uint32_t)b[0] | (uint32_t)b[1] << 8 | (uint32_t)b[2] << 16 | (uint32_t)b[3] << 24; }
static uint64_t rd_u64(FILE *f) { uint64_t lo = rd_u32(f), hi = rd_u32(f); return lo | hi << 32; }
static float    rd_f32(FILE *f) { uint32_t u = rd_u32(f); float v; memcpy(&v, &u, 4); return v; }

int main(int argc, char **argv)
{
    if (argc < 2) {
        fprintf(stderr, "usage: %s <file.vhr> [--tilt deg] [--no-gyro] [--csv path]\n", argv[0]);
        return 2;
    }
    const char *csv_path = NULL, *dump_path = NULL;
    int use_gyro = 1;
    int tilt_override = 0;
    float tilt_deg = 0.f;
    for (int i = 2; i < argc; i++) {
        if (!strcmp(argv[i], "--no-gyro")) use_gyro = 0;
        else if (!strcmp(argv[i], "--tilt") && i + 1 < argc) { tilt_deg = (float)atof(argv[++i]); tilt_override = 1; }
        else if (!strcmp(argv[i], "--csv") && i + 1 < argc) csv_path = argv[++i];
        else if (!strcmp(argv[i], "--dump") && i + 1 < argc) dump_path = argv[++i];
        else { fprintf(stderr, "unknown arg %s\n", argv[i]); return 2; }
    }
#ifndef VH_DEBUG_TRACKS
    if (dump_path) { fprintf(stderr, "--dump requires a -DVH_DEBUG_TRACKS build (make build/replay_dump)\n"); return 2; }
#endif

    FILE *f = fopen(argv[1], "rb");
    if (!f) { perror(argv[1]); return 1; }
    if (rd_u32(f) != MAGIC || rd_u32(f) != 1) { fprintf(stderr, "bad header\n"); return 1; }
    uint16_t w = rd_u16(f), h = rd_u16(f);
    float fx = rd_f32(f), fy = rd_f32(f), cx = rd_f32(f), cy = rd_f32(f);
    float file_tilt = rd_f32(f);
    uint32_t n_gyro = rd_u32(f), n_frames = rd_u32(f);
    if (!tilt_override) tilt_deg = file_tilt;
    if (w != VH_IMG_W || h != VH_IMG_H) {
        fprintf(stderr, "file is %ux%u but library built for %ux%u\n", w, h, VH_IMG_W, VH_IMG_H);
        return 1;
    }

    gyro_rec *gyro = malloc(n_gyro * sizeof *gyro);
    for (uint32_t i = 0; i < n_gyro; i++) {
        gyro[i].t_us = rd_u64(f);
        gyro[i].wx = rd_f32(f); gyro[i].wy = rd_f32(f); gyro[i].wz = rd_f32(f);
    }

    /* body (DCS native: x fwd, y up, z right) -> camera (x right, y down,
     * z forward), camera pitched up by tilt about the body right axis:
     *   cam_x = body_z
     *   cam_y = sin(t)*body_x - cos(t)*body_y
     *   cam_z = cos(t)*body_x + sin(t)*body_y            */
    float t = tilt_deg * 3.14159265358979f / 180.f;
    vh_params prm = {
        .cam = { .fx = fx, .fy = fy, .cx = cx, .cy = cy },
        .r_cb = { 0.f,          0.f,           1.f,
                  sinf(t),      -cosf(t),      0.f,
                  cosf(t),      sinf(t),       0.f },
        .auto_rekey = true,
    };

    static vh_ctx ctx; /* ~330 KB: keep off the stack */
    vh_init(&ctx, &prm);

    FILE *csv = csv_path ? fopen(csv_path, "w") : NULL;
    if (csv) fprintf(csv, "frame,t_us,status,res_x_px,res_y_px,divergence_px,n_tracked,n_keyframe,rekeyed\n");
    FILE *dump = dump_path ? fopen(dump_path, "w") : NULL;
    (void)dump;

    static uint8_t buf[VH_IMG_W * VH_IMG_H];
    vh_image img = { .data = buf, .w = VH_IMG_W, .h = VH_IMG_H, .stride = VH_IMG_W };

    uint32_t gi = 0;
    int n_ok = 0, n_deg = 0, n_lost = 0, n_rekey = 0;
    double sum_absx = 0, sum_absy = 0, sum_absdiv = 0;
    long tracked_total = 0;

    printf("%s: %u frames, %u gyro samples, fx=%.1f tilt=%.1f%s\n",
           argv[1], n_frames, n_gyro, fx, tilt_deg, use_gyro ? "" : "  [GYRO DISABLED]");

    for (uint32_t i = 0; i < n_frames; i++) {
        uint64_t t_us = rd_u64(f);
        if (fread(buf, 1, sizeof buf, f) != sizeof buf) { fprintf(stderr, "truncated file\n"); return 1; }

        if (use_gyro)
            for (; gi < n_gyro && gyro[gi].t_us <= t_us; gi++)
                vh_gyro(&ctx, gyro[gi].t_us, gyro[gi].wx, gyro[gi].wy, gyro[gi].wz);

        if (i == 0) {
            int n = vh_set_keyframe(&ctx, &img, t_us);
            printf("frame %4u: keyframe with %d features\n", i, n);
            continue;
        }

        vh_result r = vh_process_frame(&ctx, &img, t_us);
        const char *st = r.status == VH_STATUS_OK ? "OK  " :
                         r.status == VH_STATUS_DEGRADED ? "DEG " :
                         r.status == VH_STATUS_LOST ? "LOST" : "NOKF";
        printf("frame %4u: %s res=(%7.2f,%7.2f) div=%6.2f tracked=%2d/%2d%s\n",
               i, st, (double)r.res_x_px, (double)r.res_y_px, (double)r.divergence_px,
               r.n_tracked, r.n_keyframe, r.rekeyed ? "  [re-key]" : "");
        if (csv) fprintf(csv, "%u,%llu,%d,%.4f,%.4f,%.4f,%d,%d,%d\n",
                         i, (unsigned long long)r.t_us, (int)r.status,
                         (double)r.res_x_px, (double)r.res_y_px, (double)r.divergence_px,
                         r.n_tracked, r.n_keyframe, (int)r.rekeyed);
#ifdef VH_DEBUG_TRACKS
        if (dump) {
            fprintf(dump, "{\"i\":%u,\"st\":%d,\"rx\":%.3f,\"ry\":%.3f,\"dv\":%.3f,"
                          "\"nt\":%d,\"nk\":%d,\"rk\":%d,\"f\":[",
                    i, (int)r.status, (double)r.res_x_px, (double)r.res_y_px,
                    (double)r.divergence_px, r.n_tracked, r.n_keyframe, (int)r.rekeyed);
            for (int k = 0; k < ctx.n_key; k++) {
                if (k) fputc(',', dump);
                /* entry: [key_x, key_y]               feature not evaluated
                 *        [key_x, key_y, px, py]       predicted, track failed
                 *        [key_x, key_y, px, py, tx, ty] tracked            */
                if (r.rekeyed || ctx.dbg_state[k] == 0)
                    fprintf(dump, "[%.1f,%.1f]", (double)ctx.fx[k], (double)ctx.fy[k]);
                else if (ctx.dbg_state[k] == 2)
                    fprintf(dump, "[%.1f,%.1f,%.2f,%.2f,%.2f,%.2f]",
                            (double)ctx.fx[k], (double)ctx.fy[k],
                            (double)ctx.dbg_pred[k][0], (double)ctx.dbg_pred[k][1],
                            (double)ctx.dbg_trk[k][0], (double)ctx.dbg_trk[k][1]);
                else
                    fprintf(dump, "[%.1f,%.1f,%.2f,%.2f]",
                            (double)ctx.fx[k], (double)ctx.fy[k],
                            (double)ctx.dbg_pred[k][0], (double)ctx.dbg_pred[k][1]);
            }
            fprintf(dump, "]}\n");
        }
#endif

        if (r.status == VH_STATUS_OK) n_ok++;
        else if (r.status == VH_STATUS_DEGRADED) n_deg++;
        else if (r.status == VH_STATUS_LOST) n_lost++;
        if (r.rekeyed) n_rekey++;
        if (r.status == VH_STATUS_OK || r.status == VH_STATUS_DEGRADED) {
            sum_absx += fabs((double)r.res_x_px);
            sum_absy += fabs((double)r.res_y_px);
            sum_absdiv += fabs((double)r.divergence_px);
            tracked_total += r.n_tracked;
        }
    }
    int n_valid = n_ok + n_deg;
    printf("\nsummary: %d ok, %d degraded, %d lost, %d re-keys\n", n_ok, n_deg, n_lost, n_rekey);
    if (n_valid)
        printf("mean |res| = (%.2f, %.2f) px, mean |div| = %.2f px, mean tracked = %.1f\n",
               sum_absx / n_valid, sum_absy / n_valid, sum_absdiv / n_valid,
               (double)tracked_total / n_valid);
    if (csv) fclose(csv);
    free(gyro);
    fclose(f);
    return 0;
}
