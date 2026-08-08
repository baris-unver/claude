/*
 * vh_rotcomp.h — gyro integration and rotation-compensated feature prediction.
 *
 * Maintains the incremental rotation of the camera since the keyframe was
 * captured, from timestamped body-rate gyro samples. Used to predict where a
 * keyframe feature should appear in the current frame if the drone had only
 * rotated; any tracked deviation from that prediction is translation-induced
 * image motion — the position-hold error signal.
 *
 * Conventions:
 *  - gyro samples are body-frame angular rates [rad/s], timestamped in us
 *  - r_cb rotates vectors from body frame to camera frame (row-major 3x3)
 *  - camera frame: x right, y down, z forward (optical axis)
 */
#ifndef VH_ROTCOMP_H
#define VH_ROTCOMP_H

#include "vh_types.h"

typedef struct {
    uint64_t t_us;
    float w[3]; /* camera-frame rate, rad/s (already mapped through r_cb) */
} vh_gyro_sample;

typedef struct {
    vh_quat q;                 /* keyframe->current camera rotation */
    uint64_t q_t_us;           /* timestamp q is integrated up to */
    vh_gyro_sample buf[VH_GYRO_BUF_LEN];
    uint32_t head, tail;       /* ring buffer: head = write, tail = read */
    float r_cb[9];
    float last_w[3];           /* most recent integrated rate (camera frame),
                                  used to extrapolate to frame timestamps that
                                  fall between gyro samples */
    float bias_c[3];           /* camera-frame gyro bias estimate, rad/s;
                                  subtracted from each mapped sample */
    bool have_time;
} vh_rotcomp;

void vh_rot_init(vh_rotcomp *rc, const float r_cb[9]);

/* Push one gyro sample (body frame). Call from the IMU driver/ISR context;
 * samples must arrive in non-decreasing timestamp order. */
void vh_rot_push(vh_rotcomp *rc, uint64_t t_us, float wx, float wy, float wz);

/* Set the camera-frame gyro bias subtracted from subsequent samples
 * (e.g. from vh_bias). Applies to samples pushed after the call. */
void vh_rot_set_bias(vh_rotcomp *rc, const float bias_c[3]);

/* Mark "now" (a frame capture time) as the new keyframe epoch: resets the
 * accumulated rotation to identity and drops older buffered samples. */
void vh_rot_rekey(vh_rotcomp *rc, uint64_t t_us);

/* Integrate buffered samples up to time t_us (a frame capture timestamp).
 * Samples newer than t_us stay buffered for the next call. */
void vh_rot_integrate_to(vh_rotcomp *rc, uint64_t t_us);

/* Apply the inverse accumulated rotation: bearing in keyframe camera frame
 * -> bearing in current camera frame. */
vh_vec3 vh_rot_key_to_cur(const vh_rotcomp *rc, vh_vec3 bearing_key);

/* --- camera model helpers (radtan distortion) --- */

/* Distorted pixel -> unit-norm-z bearing (x, y, 1) in the camera frame. */
vh_vec3 vh_cam_pixel_to_bearing(const vh_camera *cam, float u, float v);

/* Camera-frame bearing -> distorted pixel. Returns false if the bearing
 * points away from the image plane (z too small). */
bool vh_cam_bearing_to_pixel(const vh_camera *cam, vh_vec3 b, float *u, float *v);

#endif /* VH_ROTCOMP_H */
