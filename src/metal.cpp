#include "include/material.hpp"

CUDA_DEVICE
bool metal::scatter(const ray &r,
		    const hit_record &rec,
		    vec3 &attenuation,
		    ray &scattered
#ifdef __CUDACC__
		    , curandState *rand
#endif
		    ) const {
	vec3 reflected = reflect(vec3::unit_vector(r.direction()), rec.normal);
	scattered = ray(rec.p, reflected + fuzz * random_in_unit_sphere(
#ifdef __CUDACC__
			rand
#endif
	));
	attenuation = albedo;
	return vec3::dot(scattered.direction(), rec.normal) > 0;
}
