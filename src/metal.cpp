#include "include/material.hpp"

bool metal::scatter(const ray &r,
		    const hit_record &rec,
		    vec3 &attenuation,
		    ray &scattered) const {
	vec3 reflected = reflect(vec3::unit_vector(r.direction()), rec.normal);
	scattered = ray(rec.p, reflected + fuzz * random_in_unit_sphere());
	attenuation = albedo;
	return vec3::dot(scattered.direction(), rec.normal) > 0;
}
