# vio_hold — keyframe-anchored visual position hold for MCU-class hardware

A small, dependency-free C11 library that lets an FPV drone **hold position
without GNSS** using only a forward-facing camera, a gyro, and (via the flight
controller) a barometer for altitude. Designed to run on an STM32N6-class
companion computer: static allocation only, no heap, no OS, float math sized
for a Cortex-M55 FPU.

## How it works

Full VIO estimates a trajectory; position *hold* only needs to detect "the
drone moved away from where it was" and push back. That reduces to
keyframe-anchored visual servoing:

1. When hold engages (drone stable), a **keyframe** is captured: up to 96
   FAST-9 corners, bucketed on a grid so they cover the whole frame.
2. Every frame, each keyframe feature's current position is **predicted
   assuming pure rotation**, using the gyro integrated from the keyframe
   epoch to the frame's capture timestamp (through the body→camera extrinsic
   and the lens model).
3. The feature is tracked from its *keyframe* template with iterative
   **pyramidal Lucas–Kanade**, warm-started at the prediction.
4. `residual = tracked − predicted` is translation-induced image motion.
   The per-frame output is the **median residual** (x, y) plus a **radial
   divergence** term (scene expansion = forward motion).

Because every frame is compared against the same keyframe template, the error
signal does not integrate or accumulate: hover is drift-free until a re-key,
and tracking against the original template also prevents feature drift.
The residual is up-to-scale (gain depends on scene depth) — that is fine for
a hold loop, which only needs the error's direction and a consistent gain.

```
IMU ISR ──► vh_gyro() ─────────────┐
                                   ▼
camera ──► vh_process_frame() ── rotation-compensated ──► median residual ──► PD ──► FC
frame        (FAST + pyr-LK)       prediction                (x, y, div)          (angle mode)
                 ▲
        vh_set_keyframe()  (on hold engage / track loss)
```

## Layout

```
include/vio_hold/   public headers (vh_hold.h is the top-level API)
src/                vh_fast.c vh_pyramid.c vh_klt.c vh_rotcomp.c vh_hold.c
examples/           stm32n6_main.c — integration skeleton (board hooks stubbed)
tests/              host-side numerical tests (synthetic scenes)
```

## Quick start (host tests)

```
make test
```

The harness synthesizes textured scenes and verifies: FAST detection, the
radtan distort/undistort round trip, exact recovery of injected sub-pixel
translations, cancellation of a known rotation fed through the gyro path, and
that the same rotation *leaks* into the residual when the gyro is not fed
(proving the compensation does real work).

## API in five lines

```c
vh_ctx ctx;                       /* ~330 KB — place in AXISRAM, not DTCM */
vh_init(&ctx, &params);           /* camera intrinsics + body→camera R */
vh_gyro(&ctx, t_us, wx, wy, wz);  /* every IMU sample, body rad/s */
vh_set_keyframe(&ctx, &img, t_us);/* anchor here (on hold engage) */
vh_result r = vh_process_frame(&ctx, &img, t_us);  /* per camera frame */
```

`r.res_x_px / r.res_y_px / r.divergence_px` feed a PD loop on roll/pitch
(see `examples/stm32n6_main.c`); `r.status` tells you when to level out and
re-key. All sizes/thresholds are `-D`-overridable (`vh_config.h`).

## STM32N6 notes

- **Memory:** `vh_ctx` ≈ 330 KB (keyframe copy + two pyramids + gyro ring).
  Put it in AXISRAM. QVGA Y8 frames come straight out of the DCMIPP; use a
  double buffer, 32-byte aligned, with MPU-non-cacheable or explicit D-cache
  invalidation.
- **Compute:** FAST + LK on ≤96 features at QVGA fits 30–50 Hz on the M55 at
  800 MHz in plain C; the inner loops (bilinear patch ops in `vh_klt.c`,
  circle test in `vh_fast.c`) vectorize well with Helium/CMSIS-DSP if you
  need headroom. The NPU is not needed for this pipeline.
- **Timestamps are the whole game:** stamp frames at capture in the ISR (not
  at processing) and IMU samples in the IMU ISR, from one monotonic µs timer.
  A few ms of camera–IMU misalignment visibly leaks rotation into the
  residual during fast stick moves.
- **Concurrency:** `vh_gyro()` is not safe to call from an ISR while
  `vh_process_frame()` runs — queue IMU samples in the ISR and drain them in
  the main loop (see the example).
- **Calibration:** camera intrinsics + distortion (OpenCV/Kalibr, offline)
  and the body→camera rotation are required. Calibrate gyro bias at arm time.
- **Sensor:** lock exposure/gain during hold and keep exposure short. A
  global-shutter mono sensor (OV9281-class) on the N6's CSI port is a big
  robustness upgrade over a rolling-shutter FPV camera; analog CVBS cameras
  need a decoder and fight interlacing — usable, not recommended.

## Control-loop wiring

- **Betaflight FC:** it accepts no position input, so the N6 acts as the
  pilot: send stick-equivalent corrections over MSP (`MSP_SET_RAW_RC`) at
  ~50 Hz with the FC in angle mode + baro altitude hold. Map residual-x →
  roll, divergence → pitch; image-y is dominated by altitude, which the baro
  loop owns.
- **ArduPilot/PX4 FC:** either the same stick-injection approach, or estimate
  scale (e.g. from baro-observed excursions) and feed velocity/position into
  the EKF — more work, only worth it beyond pure hover.
- **Bench-check signs before flight**: move the drone by hand with props off
  and verify commands oppose the motion on both axes.

## Limits (by design)

- Hold stiffness scales with `fx / scene_depth`: staring at a distant horizon
  gives a soft hold. Best with texture at 2–15 m.
- Needs texture and light; on `VH_STATUS_LOST` the example levels out and
  falls back to the FC's attitude + baro hold, then re-anchors.
- Each re-key inherits whatever position offset existed at that moment
  (typically centimeters at hover). Between re-keys there is no drift.
- Yaw should be held by the FC's gyro heading loop, not by this pipeline.
