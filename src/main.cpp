#include <iostream>

int main(void) {
	int nx = 960;
	int ny = 540;

	std::cout << "P3\n" << nx << " " << ny << "\n255\n";

	for (int j = ny - 1; j >= 0; --j)
		for (int i = 0; i < nx; ++i) {
			float r = float(i) / float(nx);
			float g = float(j) / float(ny);
			float b = 1.0f;
			int ir = int(255.99 * r);
			int ig = int(255.99 * g);
			int ib = int(255.99 * b);

			std::cout << ir << " "
				  << ig << " "
				  << ib << '\n';
		}
}
