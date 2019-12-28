#ifndef _UTIL_H
#define _UTIL_H

#include <random>
#include <functional>
#include "hitable.hpp"

vec3 color(const ray& r, hitable *world) {
	hit_record rec;
	if (world->hit(r, 0.0f, MAXFLOAT, rec))
		return 0.5 * vec3(rec.normal.x() + 1.0f,
				  rec.normal.y() + 1.0f,
				  rec.normal.z() + 1.0f);
	vec3 unit_direction = vec3::unit_vector(r.direction());
	float t = 0.5f * (unit_direction.y() + 1.0f);

	return (1.0f - t) * vec3(1.0f, 1.0f, 1.0f) + t * vec3(0.5f, 0.7f, 1.0f);
}

inline double random_double() {
	static std::uniform_real_distribution<double> distrib(0.0, 1.0);
	static std::mt19937 gen;
	static std::function<double()> rng = std::bind(distrib, gen);
	return rng();
}

#endif
