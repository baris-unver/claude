#include "vio_hold/vh_fast.h"

#include <string.h>

/* Bresenham circle of radius 3, clockwise from 12 o'clock. x right, y down. */
static const int8_t CDX[16] = { 0, 1, 2, 3, 3, 3, 2, 1, 0, -1, -2, -3, -3, -3, -2, -1 };
static const int8_t CDY[16] = { -3, -3, -2, -1, 0, 1, 2, 3, 3, 3, 2, 1, 0, -1, -2, -3 };

/* True if the 16-bit ring mask contains a contiguous run of >= 9 set bits
 * (with wrap-around). Doubling the mask into 32 bits linearizes the wrap;
 * each AND-shift step shortens required runs by one. */
static inline bool has_arc9(uint32_t m16)
{
    uint32_t m = m16 | (m16 << 16);
    m &= m << 1;
    m &= m << 2;
    m &= m << 4;
    m &= m << 1; /* now bit i set <=> bits i..i-8 all set: run of 9 */
    return m != 0;
}

int vh_fast_detect(const vh_image *img, vh_corner *out)
{
    const int w = img->w, h = img->h, stride = img->stride;
    const uint8_t *px = img->data;
    const int t = VH_FAST_THRESHOLD;

    /* Precompute circle offsets in bytes for this stride. */
    int32_t off[16];
    for (int i = 0; i < 16; i++)
        off[i] = (int32_t)CDY[i] * stride + CDX[i];

    /* One best-corner slot per grid cell. */
    vh_corner best[VH_MAX_FEATURES];
    for (int i = 0; i < VH_MAX_FEATURES; i++)
        best[i].score = -1;

    const int x0 = VH_DET_MARGIN, x1 = w - VH_DET_MARGIN;
    const int y0 = VH_DET_MARGIN, y1 = h - VH_DET_MARGIN;

    for (int y = y0; y < y1; y++) {
        const uint8_t *row = px + (int32_t)y * stride;
        const int cell_row = (y * VH_GRID_ROWS) / h;
        for (int x = x0; x < x1; x++) {
            const uint8_t *p = row + x;
            const int c = *p;
            const int hi = c + t, lo = c - t;

            /* Quick reject on the 4 compass pixels: a 9-arc always covers at
             * least 2 of them, so fewer than 2 bright and 2 dark => no corner. */
            int nb = 0, nd = 0;
            for (int i = 0; i < 16; i += 4) {
                const int v = p[off[i]];
                nb += (v > hi);
                nd += (v < lo);
            }
            if (nb < 2 && nd < 2)
                continue;

            uint32_t mb = 0, md = 0;
            int32_t sb = 0, sd = 0; /* summed contrast of bright/dark pixels */
            for (int i = 0; i < 16; i++) {
                const int v = p[off[i]];
                if (v > hi) {
                    mb |= 1u << i;
                    sb += v - c;
                } else if (v < lo) {
                    md |= 1u << i;
                    sd += c - v;
                }
            }

            int32_t score;
            if (has_arc9(mb))
                score = sb;
            else if (has_arc9(md))
                score = sd;
            else
                continue;

            if (score < VH_FAST_MIN_SCORE)
                continue;

            const int cell = cell_row * VH_GRID_COLS + (x * VH_GRID_COLS) / w;
            if (score > best[cell].score) {
                best[cell].score = score;
                best[cell].x = (float)x;
                best[cell].y = (float)y;
            }
        }
    }

    int n = 0;
    for (int i = 0; i < VH_MAX_FEATURES; i++)
        if (best[i].score >= 0)
            out[n++] = best[i];
    return n;
}
