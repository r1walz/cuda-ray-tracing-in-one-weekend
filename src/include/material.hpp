#ifndef _MATERIAL_H
#define _MATERIAL_H

#include "hittable.hpp"

#ifdef __CUDACC__
#include <curand_kernel.h>
CUDA_DEVICE extern vec3 random_in_unit_sphere(curandState*);
#else
extern vec3 random_in_unit_sphere();
#endif
CUDA_DEVICE extern vec3 reflect(const vec3&, const vec3&);

class material {
public:
	CUDA_DEVICE
	virtual bool scatter(const ray &r,
			     const hit_record &rec,
			     vec3 &attenuation,
			     ray &scattered
#ifdef __CUDACC__
			     , curandState *rand
#endif
			     ) const = 0;
};

class lambertian : public material {
public:
	CUDA_DEVICE lambertian(const vec3 &a) : albedo(a) {}
	CUDA_DEVICE virtual bool scatter(const ray &r, const hit_record &rec,
					 vec3 &attenuation, ray &scattered
#ifdef __CUDACC__
			     		 , curandState *rand
#endif
					 ) const;
	vec3 albedo;
};

class metal : public material {
public:
	CUDA_DEVICE metal(const vec3 &a, float f) : albedo(a) { fuzz = f < 1 ? f : 1; }
	CUDA_DEVICE virtual bool scatter(const ray &r, const hit_record &rec,
					 vec3 &attenuation, ray &scattered
#ifdef __CUDACC__
			     		 , curandState *rand
#endif
					 ) const;
	vec3 albedo;
	float fuzz;
};

#endif
