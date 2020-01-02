#ifndef _CAMERA_H
#define _CAMERA_H

#include "ray.hpp"

class camera {
public:
	CUDA_DEVICE camera(vec3 lookfrom, vec3 lookat, vec3 vup,
			   float vfov, float aspect);
	CUDA_DEVICE ray get_ray(float u, float v);

	vec3 origin, vertical, horizontal, lower_left_corner;
};

#endif
