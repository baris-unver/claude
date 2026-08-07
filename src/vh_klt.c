#include "vio_hold/vh_klt.h"

#include <math.h>

#define HW VH_KLT_HALF_WIN
#define WIN (2 * HW + 1)
#define NPX (WIN * WIN)

/* Bilinear sample at (x, y). Caller guarantees 0 <= x < w-1, 0 <= y < h-1. */
static inline float bilerp(const vh_image *im, float x, float y)
{
    const int ix = (int)x, iy = (int)y;
    const float ax = x - (float)ix, ay = y - (float)iy;
    const uint8_t *p = im->data + (int32_t)iy * im->stride + ix;
    const float top = (float)p[0] + ax * ((float)p[1] - (float)p[0]);
    const float bot = (float)p[im->stride] + ax * ((float)p[im->stride + 1] - (float)p[im->stride]);
    return top + ay * (bot - top);
}

/* Patch (and its gradient) must fit with a 1 px interpolation apron. */
static inline bool in_bounds(const vh_image *im, float x, float y)
{
    const float m = (float)(HW + 2);
    return x >= m && y >= m && x <= (float)im->w - 1.0f - m && y <= (float)im->h - 1.0f - m;
}

bool vh_klt_track(const vh_pyramid *ref, const vh_pyramid *cur,
                  float rx, float ry, float *cx, float *cy)
{
    /* Displacement estimate carried across levels, expressed at level 0. */
    float dx = *cx - rx;
    float dy = *cy - ry;

    float tmpl[NPX], gx[NPX], gy[NPX];

    for (int l = VH_PYR_LEVELS - 1; l >= 0; l--) {
        const vh_image *ri = &ref->lvl[l];
        const vh_image *ci = &cur->lvl[l];
        const float s = 1.0f / (float)(1 << l);
        const float rlx = rx * s, rly = ry * s;

        /* Near the border a patch may not fit at coarse levels even though it
         * fits at level 0. Skip such levels (carrying the displacement down)
         * instead of dropping the feature; only level 0 is mandatory. */
        if (!in_bounds(ri, rlx, rly) || !in_bounds(ci, rlx + dx * s, rly + dy * s)) {
            if (l == 0)
                return false;
            continue;
        }

        /* Template values and gradients from the reference patch (constant
         * over iterations => the 2x2 Gauss-Newton Hessian is built once). */
        float g11 = 0.0f, g12 = 0.0f, g22 = 0.0f;
        int k = 0;
        for (int j = -HW; j <= HW; j++) {
            for (int i = -HW; i <= HW; i++, k++) {
                const float px = rlx + (float)i, py = rly + (float)j;
                tmpl[k] = bilerp(ri, px, py);
                const float ix = 0.5f * (bilerp(ri, px + 1.0f, py) - bilerp(ri, px - 1.0f, py));
                const float iy = 0.5f * (bilerp(ri, px, py + 1.0f) - bilerp(ri, px, py - 1.0f));
                gx[k] = ix;
                gy[k] = iy;
                g11 += ix * ix;
                g12 += ix * iy;
                g22 += iy * iy;
            }
        }

        const float det = g11 * g22 - g12 * g12;
        if (det < 1e-4f)
            return false; /* degenerate texture (edge/flat) */
        const float inv = 1.0f / det;

        float lx = dx * s, ly = dy * s; /* displacement at this level */

        for (int it = 0; it < VH_KLT_MAX_ITER; it++) {
            const float px0 = rlx + lx, py0 = rly + ly;
            if (!in_bounds(ci, px0, py0))
                return false;

            float b1 = 0.0f, b2 = 0.0f;
            k = 0;
            for (int j = -HW; j <= HW; j++) {
                for (int i = -HW; i <= HW; i++, k++) {
                    const float e = tmpl[k] - bilerp(ci, px0 + (float)i, py0 + (float)j);
                    b1 += gx[k] * e;
                    b2 += gy[k] * e;
                }
            }

            const float ux = inv * (g22 * b1 - g12 * b2);
            const float uy = inv * (g11 * b2 - g12 * b1);
            lx += ux;
            ly += uy;
            if (ux * ux + uy * uy < VH_KLT_EPS * VH_KLT_EPS)
                break;
        }

        dx = lx / s;
        dy = ly / s;
    }

    /* Appearance gate: mean absolute residual over the level-0 patch. */
    {
        const vh_image *ri = &ref->lvl[0];
        const vh_image *ci = &cur->lvl[0];
        const float px0 = rx + dx, py0 = ry + dy;
        if (!in_bounds(ci, px0, py0) || !in_bounds(ri, rx, ry))
            return false;
        float sum = 0.0f;
        for (int j = -HW; j <= HW; j++)
            for (int i = -HW; i <= HW; i++)
                sum += fabsf(bilerp(ri, rx + (float)i, ry + (float)j) -
                             bilerp(ci, px0 + (float)i, py0 + (float)j));
        if (sum / (float)NPX > VH_KLT_MAX_RESIDUAL)
            return false;
    }

    *cx = rx + dx;
    *cy = ry + dy;
    return true;
}
