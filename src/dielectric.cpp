#include "include/material.hpp"

CUDA_DEVICE
bool dielectric::scatter(const ray& r,
			 const hit_record &rec,
			 vec3 &attenuation,
			 ray &scattered
#ifdef __CUDACC__
			 , curandState *rand
#endif
			 ) const {
	float cosine;
	float ni_over_nt;
	float reflect_prob;
	vec3 refracted;
	vec3 outward_normal;

	vec3 reflected = reflect(r.direction(), rec.normal);
	attenuation = vec3(1.0f, 1.0f, 1.0f);

	if (vec3::dot(r.direction(), rec.normal) > 0.0f) {
		outward_normal = -rec.normal;
		ni_over_nt = ref_idx;
		cosine = ref_idx * vec3::dot(r.direction(), rec.normal)
			 / r.direction().length();
	} else {
		outward_normal = rec.normal;
		ni_over_nt = 1.0f / ref_idx;
		cosine = -vec3::dot(r.direction(), rec.normal)
			 / r.direction().length();
	}
	if (refract(r.direction(), outward_normal, ni_over_nt, refracted))
		reflect_prob = schlick(cosine, ref_idx);
	else
		reflect_prob = 1.0f;
#ifdef __CUDACC__
	if (curand_uniform(rand) < reflect_prob)
#else
	if (random_double() < reflect_prob)
#endif
		scattered = ray(rec.p, reflected);
	else
		scattered = ray(rec.p, refracted);
	return true;
}
