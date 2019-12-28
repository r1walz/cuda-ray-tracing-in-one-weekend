#ifndef _MATERIAL_H
#define _MATERIAL_H

#include "hitable.hpp"

extern vec3 random_in_unit_sphere();
extern vec3 reflect(const vec3&, const vec3&);

class material {
public:
	virtual bool scatter(const ray &r,
			     const hit_record &rec,
			     vec3 &attenuation,
			     ray &scattered) const = 0;
};

class lambertian : public material {
public:
	lambertian(const vec3 &a) : albedo(a) {}
	virtual bool scatter(const ray &r, const hit_record &rec,
			     vec3 &attenuation, ray &scattered) const;
	vec3 albedo;
};

class metal : public material {
public:
	metal(const vec3 &a, float f) : albedo(a) { fuzz = f < 1 ? f : 1; }
	virtual bool scatter(const ray &r, const hit_record &rec,
			     vec3 &attenuation, ray &scattered) const;
	vec3 albedo;
	float fuzz;
};

#endif
