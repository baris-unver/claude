/*
 * vh_fast.h — FAST-9 corner detection with grid bucketing.
 *
 * Detects FAST-9 corners (9 contiguous circle pixels all brighter or all
 * darker than center +/- threshold), scores them, and keeps the single best
 * corner per grid cell so features cover the whole frame. This replaces
 * classic non-max suppression and needs no full-frame score map.
 */
#ifndef VH_FAST_H
#define VH_FAST_H

#include "vh_types.h"

typedef struct {
    float x, y;     /* corner position, px (integer-valued after detection) */
    int32_t score;  /* arc contrast score, higher = stronger */
} vh_corner;

/*
 * Detect corners in `img`. Writes at most VH_MAX_FEATURES corners (one per
 * grid cell, strongest wins) into `out`. Returns the number written.
 * Uses VH_FAST_THRESHOLD / VH_FAST_MIN_SCORE.
 */
int vh_fast_detect(const vh_image *img, vh_corner *out);

/* Same, with runtime threshold / minimum score (for the low-contrast
 * keyframe retry and host-side experiments). */
int vh_fast_detect_ex(const vh_image *img, vh_corner *out,
                      int threshold, int32_t min_score);

#endif /* VH_FAST_H */
