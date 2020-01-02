#include <iostream>
#include "include/directives.hpp"
#include "include/util.hpp"

int main(void) {
	int nx = 960;
	int ny = 540;
	float *output = new float[nx * ny * 3];

	vec3 origin;
	vec3 vertical(0.0f, 18.0f, 0.0f);
	vec3 horizontal(32.0f, 0.0f, 0.0f);
	vec3 lower_left_corner(-16.0f, -9.0f, -9.0f);

	hittable **dlist;
	hittable **dworld;

#ifdef __CUDACC__
	float *doutput;
	int num_blocks = 4096;
	int block_size = 256;

	vec3 *dorigin, *dvert, *dhori, *dllc;
	cudaMalloc((void **)&dorigin, sizeof(vec3));
	cudaMalloc((void **)&dvert, sizeof(vec3));
	cudaMalloc((void **)&dhori, sizeof(vec3));
	cudaMalloc((void **)&dllc, sizeof(vec3));

	cudaMemcpy((void *)dorigin, (void *)&origin,
		   sizeof(vec3), cudaMemcpyHostToDevice);
	cudaMemcpy((void *)dvert, (void *)&vertical,
		   sizeof(vec3), cudaMemcpyHostToDevice);
	cudaMemcpy((void *)dhori, (void *)&horizontal,
		   sizeof(vec3), cudaMemcpyHostToDevice);
	cudaMemcpy((void *)dllc, (void *)&lower_left_corner,
		   sizeof(vec3), cudaMemcpyHostToDevice);

	cudaMalloc((void **)&dlist, 2 * sizeof(hittable *));
	cudaMalloc((void **)&dworld, sizeof(hittable *));
	initiate_world<<<1, 1>>>(dlist, dworld);

	cudaMalloc((void **)&doutput, nx * ny * 3 * sizeof(float));
	paint_pixel<<<num_blocks, block_size>>>(nx, ny, dorigin,
		   dvert, dhori, dllc, dworld, doutput);
	cudaMemcpy((void *)output, (void *)doutput,
		   nx * ny * 3 * sizeof(float), cudaMemcpyDeviceToHost);
#else
	dlist = new hittable*[2];
	dlist[0] = new sphere(vec3(0.0f, 0.0f, -1.0f), 0.5f);
	dlist[1] = new sphere(vec3(0.0f, -100.5f, -1.0f), 100.0f);
	dworld = new hittable*[1];
	*dworld = new hittable_list(dlist, 2);

	paint_pixel(nx, ny, &origin, &vertical, &horizontal,
		    &lower_left_corner, dworld, output);
#endif

	std::cout << "P3\n" << nx << " " << ny << "\n255\n";

	for (int j = ny - 1; j >= 0; --j)
		for (int i = 0; i < nx; ++i)
			std::cout << int(255.99 * output[i * ny * 3 + j * 3 ]) << " "
				  << int(255.99 * output[i * ny * 3 + j * 3 + 1]) << " "
				  << int(255.99 * output[i * ny * 3 + j * 3 + 2]) << '\n';

	delete [] output;
#ifdef __CUDACC__
	cudaFree(doutput);
	cudaFree(dorigin);
	cudaFree(dvert);
	cudaFree(dhori);
	cudaFree(dllc);
	cudaFree(dlist);
	cudaFree(dworld);
	cudaDeviceReset();
#else
	delete dlist[0];
	delete dlist[1];
	delete [] dlist;
	delete [] *dworld;
	delete [] dworld;
#endif

	return 0;
}
