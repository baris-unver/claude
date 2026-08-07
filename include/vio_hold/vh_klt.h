/*
 * vh_klt.h — iterative pyramidal Lucas-Kanade feature tracking.
 *
 * Tracks a patch from a reference pyramid (the keyframe) into a current
 * pyramid. The caller supplies an initial guess for the current position
 * (here: the gyro-rotation-predicted location plus the last frame's median
 * translation), which keeps the search well inside LK's convergence basin.
 */
#ifndef VH_KLT_H
#define VH_KLT_H

#include "vh_types.h"

/*
 * Track one feature.
 *   ref      : keyframe pyramid
 *   cur      : current-frame pyramid
 *   rx, ry   : feature position in the reference image, px (level 0)
 *   cx, cy   : in: initial guess in the current image; out: refined position
 * Returns true on success (converged, in bounds, appearance residual below
 * VH_KLT_MAX_RESIDUAL); false means the feature should be dropped this frame.
 */
bool vh_klt_track(const vh_pyramid *ref, const vh_pyramid *cur,
                  float rx, float ry, float *cx, float *cy);

#endif /* VH_KLT_H */
