#include <iostream>
#include "include/util.hpp"
#include "include/camera.hpp"
#include "include/sphere.hpp"
#include "include/material.hpp"
#include "include/hitablelist.hpp"

int main(void) {
	int nx = 960;
	int ny = 540;
	int ns = 100;

	vec3 origin;
	vec3 vertical(0.0f, 18.0f, 0.0f);
	vec3 horizontal(32.0f, 0.0f, 0.0f);
	vec3 lower_left_corner(-16.0f, -9.0f, -9.0f);

	hitable *list[5];
	list[0] = new sphere(vec3(0.0f, 0.0f, -1.0f),
			     0.5f, new lambertian(vec3(0.1f, 0.2f, 0.5f)));
	list[1] = new sphere(vec3(0.0f, -100.5f, -1.0f),
			     100.0f, new lambertian(vec3(0.8f, 0.8f, 0.0f)));
	list[2] = new sphere(vec3(1.0f, 0.0f, -1.0f),
			     0.5f, new metal(vec3(0.8f, 0.6f, 0.2f), 0.0f));
	list[3] = new sphere(vec3(-1.0f, 0.0f, -1.0f),
			     0.5f, new dielectric(1.5));
	list[4] = new sphere(vec3(-1.0f, 0.0f, -1.0f),
			     -0.45f, new dielectric(1.5));
	hitable *world = new hitable_list(list, 5);
	camera cam(vec3(-2.0f, 2.0f, 1.0f),
		   vec3(0.0f, 0.0f, -1.0f),
		   vec3(0.0f, 1.0f, 0.0f),
		   25, float(nx) / float(ny));

	std::cout << "P3\n" << nx << " " << ny << "\n255\n";

	for (int j = ny - 1; j >= 0; --j)
		for (int i = 0; i < nx; ++i) {
			vec3 col;
			for (int s = 0; s < ns; ++s) {
				float u = float(i + random_double()) / float(nx);
				float v = float(j + random_double()) / float(ny);
				ray r = cam.get_ray(u, v);
				col += color(r, world, 0);
			}
			col /= float(ns);
			col = vec3(sqrt(col[0]), sqrt(col[1]), sqrt(col[2]));
			int ir = int(255.99 * col[0]);
			int ig = int(255.99 * col[1]);
			int ib = int(255.99 * col[2]);

			std::cout << ir << " "
				  << ig << " "
				  << ib << '\n';
		}
}
