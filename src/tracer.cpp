#include <iostream>
#include "include/directives.hpp"
#include "include/util.hpp"

int main(void) {
	int nx = 960;
	int ny = 540;
	int ns = 100;
	float *output = new float[nx * ny * ns * 3];

	float aperture = 0.1f;
	float dist_to_focus = 10.0f;

	camera **dcam;
	hittable **dworld;

#ifdef __CUDACC__
	float *doutput;
	int num_blocks = 4096;
	int block_size = 256;
	curandState *rand, *drand;

	cudaMalloc((void **)&dworld, sizeof(hittable *));
	cudaMalloc((void **)&dcam, sizeof(camera *));
	cudaMalloc((void **)&rand, nx * ny * sizeof(curandState));
	cudaMalloc((void **)&drand, sizeof(curandState));
	cudaMalloc((void **)&doutput, nx * ny * ns * 3 * sizeof(float));

	init_random_item<<<1, 1>>>(drand);
	initiate_world<<<1, 1>>>(nx, ny, aperture, dist_to_focus,
				 dworld, dcam, drand);
	init_random<<<num_blocks, block_size>>>(nx, ny, rand);
	paint_pixel<<<num_blocks, block_size>>>(nx, ny, ns, dcam, dworld, rand, doutput);

	cudaMemcpy((void *)output, (void *)doutput,
		   nx * ny * ns * 3 * sizeof(float), cudaMemcpyDeviceToHost);
#else
	dworld = new hittable*[1];
	dcam = new camera*[1];
	initiate_world(nx, ny, aperture, dist_to_focus,
		       dworld, dcam);
	paint_pixel(nx, ny, ns, dcam, dworld, output);
#endif

	for (int i = 0; i < nx; ++i)
	for (int j = 0; j < ny; ++j)
	for (int k = 1; k < ns; ++k) {
		output[i * ny * ns * 3 + j * ns * 3] += output[i * ny * ns * 3 + j * ns * 3 + k * 3];
		output[i * ny * ns * 3 + j * ns * 3 + 1] += output[i * ny * ns * 3 + j * ns * 3 + k * 3 + 1];
		output[i * ny * ns * 3 + j * ns * 3 + 2] += output[i * ny * ns * 3 + j * ns * 3 + k * 3 + 2];
	}

	std::cout << "P3\n" << nx << " " << ny << "\n255\n";

	for (int j = ny - 1; j >= 0; --j)
	for (int i = 0; i < nx; ++i)
		std::cout << int(255.99 * sqrt(output[i * ny * ns * 3 + j * ns * 3] / ns)) << " "
			  << int(255.99 * sqrt(output[i * ny * ns * 3 + j * ns * 3 + 1] / ns)) << " "
			  << int(255.99 * sqrt(output[i * ny * ns * 3 + j * ns * 3 + 2] / ns)) << '\n';

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
