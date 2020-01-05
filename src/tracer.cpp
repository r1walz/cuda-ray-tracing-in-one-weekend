#include <iostream>
#include "include/directives.hpp"
#include "include/util.hpp"

#include <chrono>

struct Timer {
	std::chrono::high_resolution_clock::time_point start, end;

	Timer() {
		start = std::chrono::high_resolution_clock::now();
	}

	~Timer() {
		end = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double> duration =
				std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
		std::cerr << "Time taken " << duration.count() << "s\n";
	}
};


int main(void) {
	int nx = 960;
	int ny = 540;
	int ns = 5;
	float *output = new float[nx * ny * ns * 3];

	float aperture = 0.1f;
	float dist_to_focus = 10.0f;

	camera **dcam;
	hittable **dworld;

#ifdef __CUDACC__
	float *doutput;
	int block_size = 256;
	int num_blocks = (nx * ny * ns * 3) / block_size + block_size;
	curandState *rand, *drand;

	cudaMalloc((void **)&dworld, sizeof(hittable *));
	cudaMalloc((void **)&dcam, sizeof(camera *));
	cudaMalloc((void **)&rand, nx * ny * sizeof(curandState));
	cudaMalloc((void **)&drand, sizeof(curandState));
	cudaMalloc((void **)&doutput, nx * ny * ns * 3 * sizeof(float));

{struct Timer t;
std::cerr << "init_random_item ";
	init_random_item<<<1, 1>>>(drand);
}
{struct Timer t;
std::cerr << "initiate_world ";
	initiate_world<<<1, 1>>>(nx, ny, aperture, dist_to_focus,
				 dworld, dcam, drand);
}
{struct Timer t;
std::cerr << "init_random ";
	init_random<<<num_blocks, block_size>>>(nx, ny, rand);
}
{struct Timer t;
std::cerr << "paint_pixel ";
	paint_pixel<<<num_blocks, block_size>>>(nx, ny, ns, dcam, dworld, rand, doutput);
}

	cudaDeviceSynchronize();

{struct Timer t;
std::cerr << "calculate_avg ";
	calculate_avg<<<num_blocks, block_size>>>(nx, ny, ns, doutput);
}

	cudaMemcpy((void *)output, (void *)doutput,
		   nx * ny * ns * 3 * sizeof(float), cudaMemcpyDeviceToHost);
#else
	dworld = new hittable*[1];
	dcam = new camera*[1];
	initiate_world(nx, ny, aperture, dist_to_focus,
		       dworld, dcam);
	paint_pixel(nx, ny, ns, dcam, dworld, output);

	for (int i = 0; i < nx; ++i)
	for (int j = 0; j < ny; ++j)
	for (int k = 1; k < ns; ++k) {
		output[i * ny * ns * 3 + j * ns * 3] += output[i * ny * ns * 3 + j * ns * 3 + k * 3];
		output[i * ny * ns * 3 + j * ns * 3 + 1] += output[i * ny * ns * 3 + j * ns * 3 + k * 3 + 1];
		output[i * ny * ns * 3 + j * ns * 3 + 2] += output[i * ny * ns * 3 + j * ns * 3 + k * 3 + 2];
	}
#endif

{struct Timer t;
std::cerr << "output ";
	std::cout << "P3\n" << nx << " " << ny << "\n255\n";

	for (int j = ny - 1; j >= 0; --j)
	for (int i = 0; i < nx; ++i)
		std::cout << int(255.99 * sqrt(output[i * ny * ns * 3 + j * ns * 3] / ns)) << " "
			  << int(255.99 * sqrt(output[i * ny * ns * 3 + j * ns * 3 + 1] / ns)) << " "
			  << int(255.99 * sqrt(output[i * ny * ns * 3 + j * ns * 3 + 2] / ns)) << '\n';
}
	delete [] output;
#ifdef __CUDACC__
	cudaFree(doutput);
	cudaFree(dcam);
	cudaFree(dworld);
	cudaFree(rand);
	cudaDeviceReset();
#else
	delete [] *dcam;
	delete [] dcam;
	delete [] *dworld;
	delete [] dworld;
#endif

	return 0;
}
