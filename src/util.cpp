#include "include/util.hpp"

CUDA_GLOBAL
void paint_pixel(int nx, int ny, float *output) {
#ifdef __CUDACC__
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (int i = idx; i < nx * ny; i += stride) {
#else
	for (int i = 0; i < nx * ny; ++i) {
#endif
		output[i * 3] = float(i / ny) / float(nx);
		output[i * 3 + 1] = float(i % ny) / float(ny);
		output[i * 3 + 2] = 1.0f;
	}
}
