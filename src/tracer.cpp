#include <iostream>
#include "include/ray.hpp"

bool hit_sphere(const vec3 &center, float radius, const ray &r) {
	vec3 oc = r.origin() - center;
	float a = vec3::dot(r.direction(), r.direction());
	float b = 2.0f * vec3::dot(oc, r.direction());
	float c = vec3::dot(oc, oc) - radius * radius;
	float discriminant = b * b - 4 * a * c;

	return discriminant > 0;
}

vec3 color(const ray& r) {
	if (hit_sphere(vec3(0.0f, 0.0f, -1.0f), 0.5f, r))
		return vec3(1.0f, 0.0f, 0.0f);
	vec3 unit_direction = vec3::unit_vector(r.direction());
	float t = 0.5f * (unit_direction.y() + 1.0f);

	return (1.0f - t) * vec3(1.0f, 1.0f, 1.0f) + t * vec3(0.5f, 0.7f, 1.0f);
}

int main(void) {
	int nx = 960;
	int ny = 540;

	vec3 origin;
	vec3 vertical(0.0f, 18.0f, 0.0f);
	vec3 horizontal(32.0f, 0.0f, 0.0f);
	vec3 lower_left_corner(-16.0f, -9.0f, -9.0f);

	std::cout << "P3\n" << nx << " " << ny << "\n255\n";

	for (int j = ny - 1; j >= 0; --j)
		for (int i = 0; i < nx; ++i) {
			float u = float(i) / float(nx);
			float v = float(j) / float(ny);
			ray r(origin, lower_left_corner +
				      u * horizontal + v * vertical);
			vec3 col = color(r);
			int ir = int(255.99 * col[0]);
			int ig = int(255.99 * col[1]);
			int ib = int(255.99 * col[2]);

			std::cout << ir << " "
				  << ig << " "
				  << ib << '\n';
		}
}
