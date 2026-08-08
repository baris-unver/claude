/*
 * vh_hold.h — keyframe-anchored visual position-hold pipeline.
 *
 * Data flow per frame:
 *   1. integrate gyro to the frame's capture timestamp
 *   2. build the current image pyramid
 *   3. for each keyframe feature: predict its current pixel assuming pure
 *      rotation, then KLT-track it from the keyframe patch
 *   4. residual = tracked - predicted, robustly aggregated (median) into a
 *      2D translation error plus a radial divergence (fore/aft) term
 *
 * The output is an up-to-scale position error relative to the keyframe pose:
 * feed it to a PD loop on roll/pitch (and optionally throttle) to hold
 * position. Because tracking is always against the *same* keyframe template,
 * the error does not accumulate — hover is drift-free until a re-key.
 */
#ifndef VH_HOLD_H
#define VH_HOLD_H

#include "vh_types.h"
#include "vh_pyramid.h"
#include "vh_rotcomp.h"

typedef enum {
    VH_STATUS_NO_KEYFRAME = 0, /* nothing to track against yet */
    VH_STATUS_OK,              /* valid hold output this frame */
    VH_STATUS_DEGRADED,        /* output valid but feature count is low */
    VH_STATUS_LOST,            /* too few tracks: output invalid, re-key needed */
} vh_status;

typedef struct {
    vh_status status;
    /* Median rotation-compensated feature displacement, px, in image axes
     * (+x right, +y down). This is (minus) the apparent scene shift caused
     * by drone translation since the keyframe. Up to scale (depends on
     * scene depth): use as an error signal, not as meters. */
    float res_x_px;
    float res_y_px;
    /* Median radial displacement about the principal point, px. Positive =
     * scene expanding = drone moved forward since the keyframe. */
    float divergence_px;
    int n_tracked;   /* features tracked this frame */
    int n_keyframe;  /* features in the current keyframe */
    uint64_t t_us;   /* frame timestamp */
    uint64_t key_t_us;
    bool rekeyed;    /* true if this frame became a new keyframe */
} vh_result;

typedef struct {
    vh_camera cam;
    float r_cb[9];    /* body -> camera rotation, row-major */
    bool auto_rekey;  /* promote current frame to keyframe when track is lost */
} vh_params;

typedef struct {
    vh_params prm;
    vh_rotcomp rot;

    /* Keyframe */
    vh_pyramid key_pyr;
    uint8_t key_img[VH_IMG_W * VH_IMG_H];      /* level-0 copy */
    uint8_t key_store[VH_PYR_STORAGE_BYTES];   /* levels 1..N-1 */
    float fx[VH_MAX_FEATURES], fy[VH_MAX_FEATURES]; /* keyframe pixels */
    vh_vec3 bearing[VH_MAX_FEATURES];               /* undistorted bearings */
    bool active[VH_MAX_FEATURES];
    int n_key;
    uint64_t key_t_us;
    bool have_key;

    /* Current frame pyramid scratch */
    vh_pyramid cur_pyr;
    uint8_t cur_store[VH_PYR_STORAGE_BYTES];

    /* Warm start: last frame's median residual, added to the rotation
     * prediction as the KLT initial guess. */
    float last_res_x, last_res_y;

#ifdef VH_DEBUG_TRACKS
    /* Host-side instrumentation for offline visualization (never define on
     * firmware builds). Per keyframe feature, state from the last
     * vh_process_frame: 0 = inactive/skipped, 1 = track failed, 2 = tracked. */
    uint8_t dbg_state[VH_MAX_FEATURES];
    float dbg_pred[VH_MAX_FEATURES][2]; /* rotation-only prediction, px */
    float dbg_trk[VH_MAX_FEATURES][2];  /* KLT-tracked position, px */
#endif
} vh_ctx;

void vh_init(vh_ctx *ctx, const vh_params *prm);

/* Feed a gyro sample (body frame, rad/s). Timestamps must be monotonic.
 * NOTE: not ISR-safe against a concurrent vh_process_frame/vh_set_keyframe —
 * either call everything from one context, or buffer IMU samples in the ISR
 * and drain them into vh_gyro() from the main loop. */
void vh_gyro(vh_ctx *ctx, uint64_t t_us, float wx, float wy, float wz);

/* Capture a keyframe from `img` (VH_IMG_W x VH_IMG_H grayscale, captured at
 * t_us). Call when the drone is reasonably stable. Returns the number of
 * features detected (0 => keyframe rejected, previous one is kept). */
int vh_set_keyframe(vh_ctx *ctx, const vh_image *img, uint64_t t_us);

/* Process one camera frame. `img` must stay valid during the call only. */
vh_result vh_process_frame(vh_ctx *ctx, const vh_image *img, uint64_t t_us);

#endif /* VH_HOLD_H */
