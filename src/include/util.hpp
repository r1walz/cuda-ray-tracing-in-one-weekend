#ifndef _UTIL_H
#define _UTIL_H

#include <random>
#include <functional>
#include "sphere.hpp"
#include "hitable.hpp"
#include "material.hpp"
#include "hitablelist.hpp"

inline double random_double() {
	static std::uniform_real_distribution<double> distrib(0.0, 1.0);
	static std::mt19937 gen;
	static std::function<double()> rng = std::bind(distrib, gen);
	return rng();
}

vec3 random_in_unit_disk() {
	vec3 p;

	do {
		p = 2.0f * vec3(random_double(), random_double(), 0.0f)
		    - vec3(1.0f, 1.0f, 0.0f);
	} while (p.squared_length() >= 1.0f);

	return p;
}

vec3 random_in_unit_sphere() {
	vec3 p;

	do {
		p = 2.0f * vec3(random_double(), random_double(),
				random_double()) - vec3(1.0f, 1.0f, 1.0f);
	} while (p.squared_length() >= 1.0f);

	return p;
}

hitable *random_scene() {
	int i = 1;
	int n = 500;
	hitable **list = new hitable*[n+1];
	list[0] =  new sphere(vec3(0.0f, -1000.0f, 0.0f),
			      1000.0f, new lambertian(vec3(0.5f, 0.5f, 0.5f)));

	for (int a = -11; a < 11; a++)
		for (int b = -11; b < 11; b++) {
			float choose_mat = random_double();
			vec3 center(a + 0.9f * random_double(),
				    0.2f, b + 0.9f * random_double());
			if ((center - vec3(4.0f, 0.2f, 0.0f)).length() > 0.9f) {
				if (choose_mat < 0.8f)
					list[i++] = new sphere( center, 0.2f,
						new lambertian(vec3(random_double() * random_double(),
								    random_double() * random_double(),
								    random_double() * random_double())));
				else if (choose_mat < 0.95f)
					list[i++] = new sphere( center, 0.2f,
							new metal(vec3(0.5 * (1.0 + random_double()),
								       0.5 * (1.0 + random_double()),
								       0.5 * (1.0 + random_double())),
								  0.5 * random_double()));
				else
					list[i++] = new sphere(center, 0.2f, new dielectric(1.5));
			}
		}

	list[i++] = new sphere(vec3(0.0f, 1.0f, 0.0f),
			       1.0f, new dielectric(1.5));
	list[i++] = new sphere(vec3(-4.0f, 1.0f, 0.0f),
			       1.0f, new lambertian(vec3(0.4f, 0.2f, 0.1f)));
	list[i++] = new sphere(vec3(4.0f, 1.0f, 0.0f),
			       1.0f, new metal(vec3(0.7f, 0.6f, 0.5f), 0.0f));

	return new hitable_list(list, i);
}

vec3 reflect(const vec3 &v, const vec3 &n) {
	return v - 2 * vec3::dot(v, n) * n;
}

bool refract(const vec3 &v, const vec3 &n, float ni_over_nt, vec3 &refracted) {
	vec3 uv = vec3::unit_vector(v);
	float dt = vec3::dot(uv, n);
	float D = 1.0f - ni_over_nt * ni_over_nt * (1 - dt * dt);

	if (D > 0.0f) {
		refracted = ni_over_nt * (uv - n * dt) - n * sqrt(D);
		return true;
	}
	return false;
}

float schlick(float cosine, float ref_idx) {
	float r0 = (1.0f - ref_idx) / (1.0f + ref_idx);
	r0 *= r0;

	return r0 + (1.0f - r0) * pow(1.0f - cosine, 5);
}

vec3 color(const ray& r, hitable *world, int depth) {
	hit_record rec;
	if (world->hit(r, 0.001f, MAXFLOAT, rec)) {
		ray scattered;
		vec3 attenuation;

		if (depth < 50 &&
		    rec.mat_ptr->scatter(r, rec, attenuation, scattered))
			return attenuation * color(scattered, world, depth + 1);
		return vec3();
	}
	vec3 unit_direction = vec3::unit_vector(r.direction());
	float t = 0.5f * (unit_direction.y() + 1.0f);

	return (1.0f - t) * vec3(1.0f, 1.0f, 1.0f) + t * vec3(0.5f, 0.7f, 1.0f);
}

#endif
