/*
 * vh_pyramid.h — 2x-downsampled image pyramid (2x2 box filter).
 */
#ifndef VH_PYRAMID_H
#define VH_PYRAMID_H

#include "vh_types.h"

/* Total bytes needed to store levels 1..N-1 of a pyramid whose level 0 is
 * W x H. Level 0 is the input image itself and is not copied. */
#define VH_PYR_STORAGE_BYTES                                              \
    (((VH_IMG_W / 2) * (VH_IMG_H / 2)) + ((VH_IMG_W / 4) * (VH_IMG_H / 4)))

/*
 * Build a pyramid over `img` (must be VH_IMG_W x VH_IMG_H). Level 0 aliases
 * `img` directly; higher levels are written into `storage`, which must hold
 * VH_PYR_STORAGE_BYTES bytes and remain valid for the pyramid's lifetime.
 */
void vh_pyr_build(vh_pyramid *pyr, const vh_image *img, uint8_t *storage);

#endif /* VH_PYRAMID_H */
