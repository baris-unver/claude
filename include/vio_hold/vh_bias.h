/*
 * vh_bias.h — online gyro-bias estimation from hold residual drift.
 *
 * PROTOTYPE. Principle: the hold residual is (up-to-scale) position error
 * plus the projection of any un-modeled rotation. Position error in an
 * actively-held hover is bounded and zero-mean; integrated gyro bias grows
 * linearly for the lifetime of the keyframe. Attributing the *persistent
 * drift rate* of the residual to rotation therefore recovers the bias:
 *
 *   theta_x ≈  res_y / fy   (rotation about camera x — pitch)
 *   theta_y ≈ -res_x / fx   (rotation about camera y — yaw-ish)
 *
 * The residual's TIME-DERIVATIVE measures the remaining bias directly
 * (constant offsets like wind lean differentiate away); an integral
 * estimator accumulates it into bias_c, fed back via vh_rot_set_bias().
 * As the estimate converges the drift stops, the observation goes to
 * zero, and the estimate holds steady.
 *
 * Observability notes:
 *  - bias about the optical axis (camera z) creates a rotational flow field
 *    whose median is ~0: it is invisible here AND equally harmless to the
 *    hold output. It is left at zero.
 *  - a constant position offset (steady wind lean) contributes res/age,
 *    which decays as the keyframe ages; with a long tau its integral is
 *    bounded and small. Guards below skip updates on fresh keyframes,
 *    large residuals, and non-OK frames.
 *  - far-field features (clouds, horizon) make this estimator exact, since
 *    their residual contains no translation at all; with ground texture it
 *    relies on the zero-mean-position argument instead.
 */
#ifndef VH_BIAS_H
#define VH_BIAS_H

#include "vh_types.h"
#include "vh_hold.h"

typedef struct {
    float tau_s;       /* time constant of each stage (default 20 s) */
    float max_res_px;  /* skip frames with |residual| above this (default 6) */
    float bias_c[3];   /* OUTPUT: smoothed camera-frame bias estimate, rad/s */
    float raw[3];      /* internal integrator state (rides hover motion) */
    float prev_res_x, prev_res_y;
    bool have_prev;
    uint64_t last_t_us;
} vh_bias;

void vh_bias_init(vh_bias *b, float tau_s);

/* Feed one hold result (call after every vh_process_frame). Updates the
 * estimate on usable frames and returns true when it did. Push the estimate
 * to the pipeline with vh_rot_set_bias(&ctx->rot, b->bias_c). */
bool vh_bias_update(vh_bias *b, const vh_result *r, const vh_camera *cam);

#endif /* VH_BIAS_H */
