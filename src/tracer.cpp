#include <iostream>
#include "include/util.hpp"
#include "include/sphere.hpp"
#include "include/hitablelist.hpp"

int main(void) {
	int nx = 960;
	int ny = 540;

	vec3 origin;
	vec3 vertical(0.0f, 18.0f, 0.0f);
	vec3 horizontal(32.0f, 0.0f, 0.0f);
	vec3 lower_left_corner(-16.0f, -9.0f, -9.0f);

	hitable *list[2];
	list[0] = new sphere(vec3(0.0f, 0.0f, -1.0f), 0.5f);
	list[1] = new sphere(vec3(0.0f, -100.5f, -1.0f), 100.0f);
	hitable *world = new hitable_list(list, 2);

	std::cout << "P3\n" << nx << " " << ny << "\n255\n";

	for (int j = ny - 1; j >= 0; --j)
		for (int i = 0; i < nx; ++i) {
			float u = float(i) / float(nx);
			float v = float(j) / float(ny);
			ray r(origin, lower_left_corner +
				      u * horizontal + v * vertical);
			vec3 col = color(r, world);
			int ir = int(255.99 * col[0]);
			int ig = int(255.99 * col[1]);
			int ib = int(255.99 * col[2]);

			std::cout << ir << " "
				  << ig << " "
				  << ib << '\n';
		}
}
