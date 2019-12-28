#ifndef _SPHERE_H
#define _SPHERE_H

#include "hitable.hpp"

class sphere : public hitable {
public:
	sphere() {}
	sphere(vec3 cen, float r) : center(cen), radius(r) {};
	virtual bool hit(const ray &r,
			 float t_min,
			 float t_max,
			 hit_record &rec) const;
	vec3 center;
	float radius;
};

#endif