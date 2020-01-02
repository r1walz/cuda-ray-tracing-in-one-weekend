#include "include/material.hpp"

CUDA_DEVICE
bool lambertian::scatter(const ray&,
			 const hit_record &rec,
			 vec3 &attenuation,
			 ray &scattered
#ifdef __CUDACC__
			 , curandState *rand
#endif
			 ) const {
	vec3 target = rec.p + rec.normal+ random_in_unit_sphere(
#ifdef __CUDACC__
								rand
#endif
		      );
	scattered = ray(rec.p, target - rec.p);
	attenuation = albedo;

	return true;
}
