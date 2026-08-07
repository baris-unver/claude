/*
 * vh_types.h — shared plain-data types for the vio_hold pipeline.
 */
#ifndef VH_TYPES_H
#define VH_TYPES_H

#include <stdint.h>
#include <stdbool.h>
#include "vh_config.h"

/* A borrowed view of an 8-bit grayscale image. Never owns memory. */
typedef struct {
    const uint8_t *data;
    uint16_t w;
    uint16_t h;
    uint16_t stride; /* bytes per row */
} vh_image;

/* Image pyramid backed by caller/context-provided storage. */
typedef struct {
    vh_image lvl[VH_PYR_LEVELS];
} vh_pyramid;

/* Unit quaternion, Hamilton convention, rotates vectors: v' = q * v * q^-1 */
typedef struct {
    float w, x, y, z;
} vh_quat;

/* Pinhole camera with radial-tangential (radtan / plumb-bob) distortion.
 * Set k1..p2 to 0 for an ideal pinhole (e.g. pre-rectified input). */
typedef struct {
    float fx, fy, cx, cy;
    float k1, k2, p1, p2;
} vh_camera;

typedef struct {
    float x, y;
} vh_vec2;

typedef struct {
    float x, y, z;
} vh_vec3;

#endif /* VH_TYPES_H */
