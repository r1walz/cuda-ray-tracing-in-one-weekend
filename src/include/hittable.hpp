#ifndef _HITABLE_H
#define _HITABLE_H

#include "ray.hpp"

class material;

struct hit_record {
	float t;
	vec3 p;
	vec3 normal;
	material *mat_ptr;
};

class hittable {
public:
	CUDA_DEVICE
	virtual bool hit(const ray &r,
			 float t_min,
			 float t_max,
			 hit_record &rec) const = 0;
};

#endif
