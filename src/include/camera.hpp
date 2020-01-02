#ifndef _CAMERA_H
#define _CAMERA_H

#include "ray.hpp"

class camera {
public:
	CUDA_DEVICE camera() :
			origin(vec3()),
			vertical(vec3(0.0f, 18.0f, 0.0f)),
			horizontal(vec3(32.0f, 0.0f, 0.0f)),
			lower_left_corner(vec3(-16.0f, -9.0f, -9.0f)) {}
	CUDA_DEVICE ray get_ray(float u, float v);

	vec3 origin, vertical, horizontal, lower_left_corner;
};

#endif
