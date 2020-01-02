#ifndef _UTIL_H
#define _UTIL_H

#include "ray.hpp"
#include "sphere.hpp"
#include "directives.hpp"
#include "hittable_list.hpp"

CUDA_DEVICE vec3 color(const ray& r, hittable *world);
CUDA_GLOBAL void initiate_world(hittable **list, hittable **world);
CUDA_GLOBAL void paint_pixel(int nx, int ny, const vec3 *origin,
			     const vec3 *vertical, const vec3 *horizontal,
			     const vec3 *lower_left_corner,
			     hittable **world, float *output);

#endif
