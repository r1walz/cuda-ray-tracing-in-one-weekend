#ifndef _CAMERA_H
#define _CAMERA_H

#include "ray.hpp"

extern vec3 random_in_unit_disk();

class camera {
public:
	camera(vec3 lookfrom, vec3 lookat, vec3 vup,
	       float vfov, float aspect, float aperture, float focus_dist);
	ray get_ray(float s, float t);

	vec3 origin, vertical, horizontal, lower_left_corner;
	vec3 u, v, w;
	float lens_radius;
};

#endif
