/*
 * stm32n6_main.c — integration skeleton for the vio_hold pipeline on an
 * STM32N6 companion computer with a forward-facing camera.
 *
 * This file shows the wiring, not a drop-in firmware: the camera (DCMIPP),
 * IMU driver, timebase, and FC link (MSP/CRSF/MAVLink) are board-specific
 * and stubbed out below. Everything marked TODO(board) needs your HAL code.
 *
 * Suggested placement of the big buffers on the N6:
 *   - vh_ctx is ~330 KB (two pyramids + keyframe copy + gyro buffer):
 *     put it in AXISRAM (.axisram section), not DTCM.
 *   - Camera DMA target buffers: AXISRAM, 32-byte aligned, and either mark
 *     the region non-cacheable via MPU or clean/invalidate D-cache around
 *     DMA completion.
 *
 * Threading model (single core, no RTOS needed):
 *   - IMU ISR (1 kHz+): timestamp sample, push to a small lock-free queue.
 *   - Camera frame-done ISR: timestamp the frame (start-of-exposure if the
 *     sensor provides it, else frame-done minus exposure), flip the
 *     double-buffer, set a flag.
 *   - Main loop: drain IMU queue into vh_gyro(), then run vh_process_frame()
 *     on the newest complete frame, then send the control correction.
 */
#include <string.h>
#include "vio_hold/vh_hold.h"

/* ------------------------------------------------------------------ */
/* Board interface — TODO(board): implement these for your hardware.  */
/* ------------------------------------------------------------------ */

/* Monotonic microsecond timebase shared by IMU and camera timestamps.
 * On STM32: a 32-bit TIM clocked at 1 MHz, extended to 64 bit on overflow. */
extern uint64_t board_micros(void);

/* Newest complete grayscale frame, or NULL if none since last call.
 * DCMIPP can output Y8 directly (or take Y from YUV422) at 320x240 into a
 * double buffer. *t_us must be the capture timestamp recorded in the ISR. */
extern const uint8_t *board_get_frame(uint64_t *t_us);

/* Drain IMU samples queued by the IMU ISR since the last call.
 * Returns number of samples written into out[] (body rates, rad/s). */
typedef struct { uint64_t t_us; float wx, wy, wz; } imu_sample_t;
extern int board_drain_imu(imu_sample_t *out, int max);

/* True while the pilot has position-hold engaged (e.g. an RC AUX switch
 * forwarded by the FC). */
extern bool board_hold_engaged(void);

/* Send stick-equivalent corrections to the flight controller. Values are
 * normalized [-1, 1]; the FC should be in angle mode (+ baro altitude hold,
 * so this loop only handles the horizontal axes; divergence -> pitch handles
 * fore/aft). Typically MSP_SET_RAW_RC at 50 Hz toward Betaflight/iNav, or
 * MAVLink MANUAL_CONTROL / velocity setpoints toward ArduPilot/PX4. */
extern void board_send_control(float roll, float pitch);

/* ------------------------------------------------------------------ */
/* Pipeline setup                                                     */
/* ------------------------------------------------------------------ */

/* Big context: keep out of DTCM. TODO(board): define the section in your
 * linker script, or drop the attribute if default placement is AXISRAM. */
static vh_ctx g_vh; /* __attribute__((section(".axisram"))) */

/* Camera intrinsics from calibration (e.g. Kalibr or OpenCV calibrate on the
 * host with captured frames). These MUST match your sensor + lens. */
static const vh_camera CAM = {
    .fx = 265.0f, .fy = 265.0f, .cx = 160.0f, .cy = 120.0f,
    .k1 = -0.30f, .k2 = 0.09f, .p1 = 0.0f, .p2 = 0.0f,
};

/* Body -> camera rotation. Example: camera looking forward along body +X,
 * camera x = body -Y (right), camera y = body -Z (down), camera z = body +X.
 * Rows are camera axes expressed in body coordinates. */
static const float R_CB[9] = {
    0.0f, -1.0f, 0.0f,
    0.0f, 0.0f, -1.0f,
    1.0f, 0.0f, 0.0f,
};

/* PD gains on the pixel-space error. Start small; effective loop gain scales
 * with fx/scene-depth, so tune at your typical operating distance. */
#define KP_PIX 0.004f  /* stick units per px */
#define KD_PIX 0.02f   /* stick units per px/s */
#define CMD_LIMIT 0.35f

static float clampf(float v, float lim)
{
    return v > lim ? lim : (v < -lim ? -lim : v);
}

int main(void)
{
    /* TODO(board): clocks, MPU/cache config, DCMIPP + sensor bring-up
     * (Y8 @ 320x240, fixed exposure/gain while holding), IMU config
     * (gyro >= 1 kHz, calibrate bias at rest before takeoff), FC link. */

    vh_params prm;
    memset(&prm, 0, sizeof(prm));
    prm.cam = CAM;
    memcpy(prm.r_cb, R_CB, sizeof(R_CB));
    prm.auto_rekey = true; /* re-anchor immediately if tracking is lost */
    vh_init(&g_vh, &prm);

    float prev_ex = 0.0f, prev_ey = 0.0f;
    uint64_t prev_t = 0;
    bool was_engaged = false;

    for (;;) {
        /* 1. IMU first, so rotation state is current before the frame. */
        imu_sample_t imu[64];
        const int n = board_drain_imu(imu, 64);
        for (int i = 0; i < n; i++)
            vh_gyro(&g_vh, imu[i].t_us, imu[i].wx, imu[i].wy, imu[i].wz);

        /* 2. Process the newest frame, if any. */
        uint64_t t_us;
        const uint8_t *frame = board_get_frame(&t_us);
        if (frame == NULL)
            continue;

        const bool engaged = board_hold_engaged();
        vh_image img = { frame, VH_IMG_W, VH_IMG_H, VH_IMG_W };

        if (engaged && !was_engaged) {
            /* Hold just engaged: anchor here. */
            vh_set_keyframe(&g_vh, &img, t_us);
            prev_ex = prev_ey = 0.0f;
            prev_t = 0;
        }
        was_engaged = engaged;
        if (!engaged) {
            board_send_control(0.0f, 0.0f);
            continue;
        }

        const vh_result r = vh_process_frame(&g_vh, &img, t_us);

        if (r.status == VH_STATUS_OK || r.status == VH_STATUS_DEGRADED) {
            /* Residual = apparent scene shift = minus the drone's own motion,
             * so commanding in the direction of the residual pushes the drone
             * back toward the keyframe pose. VERIFY SIGNS ON A BENCH before
             * flight: move the drone by hand and check the commands oppose
             * the motion. Camera x (image right) -> roll; divergence
             * (forward motion since keyframe) -> pitch back. Image y is
             * dominated by altitude, which the FC's baro loop owns. */
            const float ex = r.res_x_px;
            const float ef = r.divergence_px;

            float dx = 0.0f, df = 0.0f;
            if (prev_t != 0 && t_us > prev_t) {
                const float dt = (float)(t_us - prev_t) * 1e-6f;
                dx = (ex - prev_ex) / dt;
                df = (ef - prev_ey) / dt;
            }
            prev_ex = ex;
            prev_ey = ef;
            prev_t = t_us;

            const float roll = clampf(KP_PIX * ex + KD_PIX * dx, CMD_LIMIT);
            const float pitch = clampf(-(KP_PIX * ef + KD_PIX * df), CMD_LIMIT);
            board_send_control(roll, pitch);

            /* Optional: proactive re-key when DEGRADED and currently well
             * centered (small residual), to refresh the anchor before it is
             * lost entirely. */
            if (r.status == VH_STATUS_DEGRADED &&
                ex * ex + r.res_y_px * r.res_y_px < 4.0f)
                vh_set_keyframe(&g_vh, &img, t_us);
        } else {
            /* LOST / NO_KEYFRAME: never keep commanding on stale vision.
             * Level out and let the FC's attitude + baro hold take over;
             * auto_rekey re-anchors as soon as features are found. */
            board_send_control(0.0f, 0.0f);
            prev_t = 0;
        }
    }
}
