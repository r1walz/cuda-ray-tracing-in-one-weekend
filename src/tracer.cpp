#include <iostream>
#include "include/directives.hpp"
#include "include/util.hpp"

int main(void) {
	int nx = 960;
	int ny = 540;
	int ns = 100;
	float *output = new float[nx * ny * ns * 3];

	vec3 lookfrom(3.0f, 3.0f, 2.0f);
	vec3 lookat(0.0f, 0.0f, -1.0f);

	float aperture = 2.0f;
	float dist_to_focus = (lookfrom - lookat).length();

	camera **dcam;
	hittable **dlist;
	hittable **dworld;

#ifdef __CUDACC__
	float *doutput;
	int num_blocks = 4096;
	int block_size = 256;
	curandState *rand;

	cudaMalloc((void **)&dlist, 2 * sizeof(hittable *));
	cudaMalloc((void **)&dworld, sizeof(hittable *));
	cudaMalloc((void **)&dcam, sizeof(camera *));
	cudaMalloc((void **)&rand, nx * ny * sizeof(curandState));
	cudaMalloc((void **)&doutput, nx * ny * ns * 3 * sizeof(float));

	initiate_world<<<1, 1>>>(nx, ny, aperture, dist_to_focus,
				 dlist, dworld, dcam);
	init_random<<<num_blocks, block_size>>>(nx, ny, rand);
	paint_pixel<<<num_blocks, block_size>>>(nx, ny, ns, dcam, dworld, rand, doutput);

	cudaMemcpy((void *)output, (void *)doutput,
		   nx * ny * ns * 3 * sizeof(float), cudaMemcpyDeviceToHost);
#else
	dlist = new hittable*[5];
	dlist[0] = new sphere(vec3(0.0f, 0.0f, -1.0f),
			     0.5f, new lambertian(vec3(0.1f, 0.2f, 0.5f)));
	dlist[1] = new sphere(vec3(0.0f, -100.5f, -1.0f),
			     100.0f, new lambertian(vec3(0.8f, 0.8f, 0.0f)));
	dlist[2] = new sphere(vec3(1.0f, 0.0f, -1.0f),
			     0.5f, new metal(vec3(0.8f, 0.6f, 0.2f), 0.0f));
	dlist[3] = new sphere(vec3(-1.0f, 0.0f, -1.0f),
			     0.5f, new dielectric(1.5));
	dlist[4] = new sphere(vec3(-1.0f, 0.0f, -1.0f),
			      -0.45f, new dielectric(1.5));
	dworld = new hittable*[1];
	*dworld = new hittable_list(dlist, 5);
	dcam = new camera*[1];
	*dcam = new camera(lookfrom, lookat,
			   vec3(0.0f, 1.0f, 0.0f),
			   25.0f, float(nx) / float(ny),
			   aperture, dist_to_focus);

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
	cudaFree(dlist);
	cudaFree(dcam);
	cudaFree(dworld);
	cudaFree(rand);
	cudaDeviceReset();
#else
	delete [] dlist;
	delete [] *dcam;
	delete [] dcam;
	delete [] *dworld;
	delete [] dworld;
#endif

	return 0;
}
