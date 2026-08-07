/*
 * vh_config.h — compile-time configuration for the vio_hold pipeline.
 *
 * All buffer sizes are static; there is no heap use anywhere in the library.
 * Override any of these with -D flags in your build.
 */
#ifndef VH_CONFIG_H
#define VH_CONFIG_H

/* Input image geometry (grayscale, 8-bit). QVGA is the sweet spot on the
 * STM32N6: pyramids for two frames fit in ~200 KB of AXISRAM and the
 * front-end runs at 30+ Hz on the M55. */
#ifndef VH_IMG_W
#define VH_IMG_W 320
#endif
#ifndef VH_IMG_H
#define VH_IMG_H 240
#endif

/* Pyramid levels (level 0 = full res). 3 levels -> LK convergence radius of
 * roughly +/-30 px at level 0, enough for hover excursions. */
#ifndef VH_PYR_LEVELS
#define VH_PYR_LEVELS 3
#endif

/* Feature budget. Detection buckets the image into a grid and keeps the
 * best corner per cell so features are spread over the whole frame. */
#ifndef VH_GRID_COLS
#define VH_GRID_COLS 12
#endif
#ifndef VH_GRID_ROWS
#define VH_GRID_ROWS 8
#endif
#define VH_MAX_FEATURES (VH_GRID_COLS * VH_GRID_ROWS)

/* FAST-9 detector */
#ifndef VH_FAST_THRESHOLD
#define VH_FAST_THRESHOLD 20 /* intensity contrast threshold */
#endif
#ifndef VH_FAST_MIN_SCORE
#define VH_FAST_MIN_SCORE 60 /* reject weak corners (sum of arc contrast) */
#endif
#ifndef VH_DET_MARGIN
#define VH_DET_MARGIN 12 /* keep-out border for detection, px */
#endif

/* Pyramidal Lucas-Kanade tracker */
#ifndef VH_KLT_HALF_WIN
#define VH_KLT_HALF_WIN 4 /* 9x9 patch */
#endif
#ifndef VH_KLT_MAX_ITER
#define VH_KLT_MAX_ITER 12
#endif
#ifndef VH_KLT_EPS
#define VH_KLT_EPS 0.03f /* convergence threshold, px */
#endif
#ifndef VH_KLT_MAX_RESIDUAL
#define VH_KLT_MAX_RESIDUAL 22.0f /* mean |I_ref - I_cur| gate after track */
#endif

/* Gyro ring buffer. Must cover the latency between a frame's capture
 * timestamp and the moment it is processed. 512 samples = 0.5 s at 1 kHz. */
#ifndef VH_GYRO_BUF_LEN
#define VH_GYRO_BUF_LEN 512
#endif

/* Keyframe management */
#ifndef VH_MIN_TRACKED
#define VH_MIN_TRACKED 15 /* below this the hold output is flagged invalid */
#endif
#ifndef VH_REKEY_FRACTION
#define VH_REKEY_FRACTION 0.35f /* re-key when tracked/initial drops below */
#endif

#endif /* VH_CONFIG_H */
