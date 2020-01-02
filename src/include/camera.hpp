#ifndef _CAMERA_H
#define _CAMERA_H

#include "ray.hpp"

#ifdef __CUDACC__
#include <curand_kernel.h>
CUDA_DEVICE extern vec3 random_in_unit_disk(curandState *rand);
#else
CUDA_HOST extern vec3 random_in_unit_disk();
#endif

class camera {
public:
	CUDA_DEVICE camera(vec3 lookfrom, vec3 lookat, vec3 vup,
	       float vfov, float aspect, float aperture, float focus_dist);
	CUDA_DEVICE ray get_ray(float s, float t
#ifdef __CUDACC__
				, curandState *rand
#endif
				);

	vec3 origin, vertical, horizontal, lower_left_corner;
	vec3 u, v, w;
	float lens_radius;
};

#endif
