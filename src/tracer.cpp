#include <iostream>
#include "include/vec3.hpp"

int main(void) {
	int nx = 960;
	int ny = 540;

	std::cout << "P3\n" << nx << " " << ny << "\n255\n";

	for (int j = ny - 1; j >= 0; --j)
		for (int i = 0; i < nx; ++i) {
			vec3 col(float(i) / float(nx), float(j) / float(ny), 1.0f);
			int ir = int(255.99 * col[0]);
			int ig = int(255.99 * col[1]);
			int ib = int(255.99 * col[2]);

			std::cout << ir << " "
				  << ig << " "
				  << ib << '\n';
		}
}
