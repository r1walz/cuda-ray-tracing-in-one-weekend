#include <iostream>
#include "include/directives.hpp"
#include "include/util.hpp"

int main(void) {
	int nx = 960;
	int ny = 540;
	float *output = new float[nx * ny * 3];

#ifdef __CUDACC__
	float *doutput;
	int num_blocks = 4096;
	int block_size = 256;
	cudaMalloc((void **)&doutput, nx * ny * 3 * sizeof(float));
	paint_pixel<<<num_blocks, block_size>>>(nx, ny, doutput);
	cudaMemcpy((void *)output, (void *)doutput,
		   nx * ny * 3 * sizeof(float), cudaMemcpyDeviceToHost);
#else
	paint_pixel(nx, ny, output);
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
	cudaDeviceReset();
#endif

	return 0;
}
