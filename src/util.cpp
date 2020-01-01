#include "include/util.hpp"
#include "include/vec3.hpp"

CUDA_GLOBAL
void paint_pixel(int nx, int ny, float *output) {
#ifdef __CUDACC__
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (int i = idx; i < nx * ny; i += stride) {
#else
	for (int i = 0; i < nx * ny; ++i) {
#endif
		vec3 col(float(i / ny) / float(nx),
			 float(i % ny) / float(ny), 1.0f);
		output[i * 3] = col[0];
		output[i * 3 + 1] = col[1];
		output[i * 3 + 2] = col[2];
	}
}
