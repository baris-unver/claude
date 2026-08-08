#include "vio_hold/vh_bias.h"

#include <math.h>
#include <string.h>

void vh_bias_init(vh_bias *b, float tau_s)
{
    memset(b, 0, sizeof(*b));
    b->tau_s = (tau_s > 0.0f) ? tau_s : 20.0f;
    b->max_res_px = 6.0f;
}

bool vh_bias_update(vh_bias *b, const vh_result *r, const vh_camera *cam)
{
    const uint64_t prev_t = b->last_t_us;
    const float prev_rx = b->prev_res_x, prev_ry = b->prev_res_y;
    const bool had_prev = b->have_prev;

    if (r->status != VH_STATUS_OK || r->rekeyed) {
        b->have_prev = false; /* never differentiate across a re-key or gap */
        b->last_t_us = r->t_us;
        return false;
    }
    b->prev_res_x = r->res_x_px;
    b->prev_res_y = r->res_y_px;
    b->have_prev = true;
    b->last_t_us = r->t_us;

    if (!had_prev || r->t_us <= prev_t)
        return false;
    if (fabsf(r->res_x_px) > b->max_res_px || fabsf(r->res_y_px) > b->max_res_px)
        return false; /* likely genuine motion, not drift */

    /* The residual's drift rate measures the REMAINING bias directly and is
     * immune to constant offsets (wind lean) since they differentiate away.
     * Signs follow the prediction path (p_c = R(q)^T p_k, image +x right,
     * +y down, camera z forward): spurious +theta_x moves the prediction +y
     * so residual_y drifts at -fy per rad; spurious +theta_y moves it -x so
     * residual_x drifts at +fx per rad. */
    const float dt = (float)(r->t_us - prev_t) * 1e-6f;
    const float rate_x = -((r->res_y_px - prev_ry) / dt) / cam->fy;
    const float rate_y = ((r->res_x_px - prev_rx) / dt) / cam->fx;

    /* Two-stage estimator. Stage 1 integrates the observation (gain dt/tau)
     * and converges onto the true bias — but it rides the derivative of
     * genuine hover motion, which can be several times the bias signal.
     * Stage 2 low-passes stage 1 with the same tau before anything is
     * applied, attenuating that ripple quadratically while the mean passes
     * through. Zero-mean hover motion therefore cancels; the persistent
     * drift does not. */
    const float k = dt / b->tau_s;
    b->raw[0] += k * rate_x;
    b->raw[1] += k * rate_y;
    b->bias_c[0] += k * (b->raw[0] - b->bias_c[0]);
    b->bias_c[1] += k * (b->raw[1] - b->bias_c[1]);
    /* bias about the optical axis is unobservable from the median residual
     * and equally irrelevant to it; both stages stay zero there. */
    return true;
}
