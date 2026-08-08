#include "vio_hold/vh_rotcomp.h"

#include <math.h>
#include <string.h>

/* ---------- quaternion helpers ---------- */

static inline vh_quat quat_mul(vh_quat a, vh_quat b)
{
    vh_quat r;
    r.w = a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z;
    r.x = a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y;
    r.y = a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x;
    r.z = a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w;
    return r;
}

static inline void quat_normalize(vh_quat *q)
{
    const float n = sqrtf(q->w * q->w + q->x * q->x + q->y * q->y + q->z * q->z);
    if (n > 0.0f) {
        const float s = 1.0f / n;
        q->w *= s;
        q->x *= s;
        q->y *= s;
        q->z *= s;
    } else {
        q->w = 1.0f;
        q->x = q->y = q->z = 0.0f;
    }
}

/* Exact exponential map: rotation vector (axis*angle) -> quaternion. */
static inline vh_quat quat_from_rotvec(float rx, float ry, float rz)
{
    const float a2 = rx * rx + ry * ry + rz * rz;
    vh_quat q;
    if (a2 < 1e-12f) {
        q.w = 1.0f;
        q.x = 0.5f * rx;
        q.y = 0.5f * ry;
        q.z = 0.5f * rz;
    } else {
        const float a = sqrtf(a2);
        const float s = sinf(0.5f * a) / a;
        q.w = cosf(0.5f * a);
        q.x = s * rx;
        q.y = s * ry;
        q.z = s * rz;
    }
    return q;
}

/* Rotate v by the conjugate (inverse) of q. */
static inline vh_vec3 quat_rotate_inv(vh_quat q, vh_vec3 v)
{
    /* v' = q^-1 * v * q, expanded via the cross-product form with -q.xyz */
    const float qx = -q.x, qy = -q.y, qz = -q.z, qw = q.w;
    const float tx = 2.0f * (qy * v.z - qz * v.y);
    const float ty = 2.0f * (qz * v.x - qx * v.z);
    const float tz = 2.0f * (qx * v.y - qy * v.x);
    vh_vec3 r;
    r.x = v.x + qw * tx + (qy * tz - qz * ty);
    r.y = v.y + qw * ty + (qz * tx - qx * tz);
    r.z = v.z + qw * tz + (qx * ty - qy * tx);
    return r;
}

/* ---------- gyro ring buffer + integration ---------- */

void vh_rot_init(vh_rotcomp *rc, const float r_cb[9])
{
    memset(rc, 0, sizeof(*rc));
    rc->q.w = 1.0f;
    memcpy(rc->r_cb, r_cb, sizeof(rc->r_cb));
}

void vh_rot_push(vh_rotcomp *rc, uint64_t t_us, float wx, float wy, float wz)
{
    const uint32_t next = (rc->head + 1u) % VH_GYRO_BUF_LEN;
    if (next == rc->tail)
        rc->tail = (rc->tail + 1u) % VH_GYRO_BUF_LEN; /* overwrite oldest */

    vh_gyro_sample *s = &rc->buf[rc->head];
    s->t_us = t_us;
    const float *R = rc->r_cb;
    s->w[0] = R[0] * wx + R[1] * wy + R[2] * wz - rc->bias_c[0];
    s->w[1] = R[3] * wx + R[4] * wy + R[5] * wz - rc->bias_c[1];
    s->w[2] = R[6] * wx + R[7] * wy + R[8] * wz - rc->bias_c[2];
    rc->head = next;
}

void vh_rot_set_bias(vh_rotcomp *rc, const float bias_c[3])
{
    rc->bias_c[0] = bias_c[0];
    rc->bias_c[1] = bias_c[1];
    rc->bias_c[2] = bias_c[2];
}

void vh_rot_rekey(vh_rotcomp *rc, uint64_t t_us)
{
    rc->q.w = 1.0f;
    rc->q.x = rc->q.y = rc->q.z = 0.0f;
    rc->q_t_us = t_us;
    rc->have_time = true;
    /* Drop samples at or before the new epoch. */
    while (rc->tail != rc->head && rc->buf[rc->tail].t_us <= t_us)
        rc->tail = (rc->tail + 1u) % VH_GYRO_BUF_LEN;
}

void vh_rot_integrate_to(vh_rotcomp *rc, uint64_t t_us)
{
    if (!rc->have_time) {
        rc->q_t_us = t_us;
        rc->have_time = true;
        return;
    }

    while (rc->tail != rc->head) {
        const vh_gyro_sample *s = &rc->buf[rc->tail];
        if (s->t_us > t_us)
            break;
        const float dt = (float)(s->t_us - rc->q_t_us) * 1e-6f;
        if (dt > 0.0f) {
            const vh_quat dq =
                quat_from_rotvec(s->w[0] * dt, s->w[1] * dt, s->w[2] * dt);
            rc->q = quat_mul(rc->q, dq);
        }
        rc->q_t_us = s->t_us;
        rc->last_w[0] = s->w[0];
        rc->last_w[1] = s->w[1];
        rc->last_w[2] = s->w[2];
        rc->tail = (rc->tail + 1u) % VH_GYRO_BUF_LEN;
    }
    if (t_us > rc->q_t_us) {
        /* Frame timestamp falls between gyro samples: extrapolate with the
         * last known rate (zero-order hold). Dropping this slice instead
         * would systematically under-rotate q by up to one IMU period per
         * frame — a bias correlated with the motion itself, which
         * accumulates for the lifetime of the keyframe. */
        const float dt = (float)(t_us - rc->q_t_us) * 1e-6f;
        const vh_quat dq = quat_from_rotvec(rc->last_w[0] * dt,
                                            rc->last_w[1] * dt,
                                            rc->last_w[2] * dt);
        rc->q = quat_mul(rc->q, dq);
        rc->q_t_us = t_us;
    }
    quat_normalize(&rc->q);
}

vh_vec3 vh_rot_key_to_cur(const vh_rotcomp *rc, vh_vec3 bearing_key)
{
    /* q integrates camera rates from the key epoch, so R(q) maps the key
     * camera frame onto the current one: R_wc = R_wk * R(q). A fixed world
     * direction p_k in key coordinates is p_c = R(q)^T p_k. */
    return quat_rotate_inv(rc->q, bearing_key);
}

/* ---------- camera model ---------- */

static inline void distort(const vh_camera *c, float x, float y, float *xd, float *yd)
{
    const float r2 = x * x + y * y;
    const float radial = 1.0f + r2 * (c->k1 + r2 * c->k2);
    *xd = x * radial + 2.0f * c->p1 * x * y + c->p2 * (r2 + 2.0f * x * x);
    *yd = y * radial + c->p1 * (r2 + 2.0f * y * y) + 2.0f * c->p2 * x * y;
}

vh_vec3 vh_cam_pixel_to_bearing(const vh_camera *cam, float u, float v)
{
    const float xd = (u - cam->cx) / cam->fx;
    const float yd = (v - cam->cy) / cam->fy;

    /* Iterative undistortion (fixed point on the radtan model). Converges in
     * a handful of iterations for moderate FOV lenses. */
    float x = xd, y = yd;
    for (int i = 0; i < 8; i++) {
        float ex, ey;
        distort(cam, x, y, &ex, &ey);
        x += xd - ex;
        y += yd - ey;
    }

    vh_vec3 b = { x, y, 1.0f };
    return b;
}

bool vh_cam_bearing_to_pixel(const vh_camera *cam, vh_vec3 b, float *u, float *v)
{
    if (b.z < 1e-3f)
        return false;
    float xd, yd;
    distort(cam, b.x / b.z, b.y / b.z, &xd, &yd);
    *u = cam->fx * xd + cam->cx;
    *v = cam->fy * yd + cam->cy;
    return true;
}
