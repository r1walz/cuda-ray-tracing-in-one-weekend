#ifndef _SPHERE_H
#define _SPHERE_H

#include "hittable.hpp"

class sphere : public hittable {
public:
	CUDA_DEVICE sphere() {}
	CUDA_DEVICE sphere(vec3 cen, float r, material *m) :
		center(cen), radius(r), mat_ptr(m) {}
	CUDA_DEVICE
	virtual bool hit(const ray &r,
			 float t_min,
			 float t_max,
			 hit_record &rec) const;
	vec3 center;
	float radius;
	material *mat_ptr;
};

#endif