#include "vio_hold/vh_pyramid.h"

void vh_pyr_build(vh_pyramid *pyr, const vh_image *img, uint8_t *storage)
{
    pyr->lvl[0] = *img;

    uint8_t *dst = storage;
    for (int l = 1; l < VH_PYR_LEVELS; l++) {
        const vh_image *src = &pyr->lvl[l - 1];
        const int dw = src->w / 2, dh = src->h / 2;

        vh_image *d = &pyr->lvl[l];
        d->data = dst;
        d->w = (uint16_t)dw;
        d->h = (uint16_t)dh;
        d->stride = (uint16_t)dw;

        for (int y = 0; y < dh; y++) {
            const uint8_t *r0 = src->data + (int32_t)(2 * y) * src->stride;
            const uint8_t *r1 = r0 + src->stride;
            uint8_t *o = dst + (int32_t)y * dw;
            for (int x = 0; x < dw; x++) {
                const int sx = 2 * x;
                o[x] = (uint8_t)((r0[sx] + r0[sx + 1] + r1[sx] + r1[sx + 1] + 2) >> 2);
            }
        }
        dst += (int32_t)dw * dh;
    }
}
