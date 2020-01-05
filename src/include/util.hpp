#ifndef _UTIL_H
#define _UTIL_H

#include <random>
#include <functional>
#include "ray.hpp"
#include "camera.hpp"
#include "sphere.hpp"
#include "material.hpp"
#include "directives.hpp"
#include "hittable_list.hpp"

#ifdef __CUDACC__
#include <curand_kernel.h>
CUDA_DEVICE vec3 random_in_unit_disk(curandState *rand);
CUDA_DEVICE vec3 random_in_unit_sphere(curandState *rand);
CUDA_DEVICE vec3 color(const ray &r, hittable *world, curandState *rand);

CUDA_GLOBAL void init_random_item(curandState *rand);
CUDA_GLOBAL void init_random(int nx, int ny, curandState *rand);
CUDA_GLOBAL void calculate_avg(int nx, int ny, int ns, float *output);
CUDA_GLOBAL void paint_pixel(int nx, int ny, int ns, camera **cam,
			     hittable **world, curandState *rand, float *output);
#endif

CUDA_HOST vec3 color(const ray& r, hittable *world);
CUDA_HOST double random_double();
CUDA_HOST vec3 random_in_unit_disk();
CUDA_HOST vec3 random_in_unit_sphere();

CUDA_DEVICE vec3 reflect(const vec3 &v, const vec3 &n);
CUDA_DEVICE bool refract(const vec3 &v, const vec3 &n,
			 float ni_over_nt, vec3 &refracted);
CUDA_DEVICE float schlick(float cosine, float ref_idx);

CUDA_GLOBAL void initiate_world(int nx, int ny, float aperture, float dist_to_focus,
				hittable **world, camera **cam
#ifdef __CUDACC__
				, curandState *rand
#endif
				);
CUDA_GLOBAL void paint_pixel(int nx, int ny, int ns, camera **cam,
			     hittable **world, float *output);

#endif
