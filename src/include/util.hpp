#ifndef _UTIL_H
#define _UTIL_H

#include "ray.hpp"
#include "directives.hpp"

CUDA_DEVICE vec3 color(const ray& r);
CUDA_GLOBAL void paint_pixel(int nx, int ny, const vec3 *origin,
			     const vec3 *vertical, const vec3 *horizontal,
			     const vec3 *lower_left_corner, float *output);

#endif
