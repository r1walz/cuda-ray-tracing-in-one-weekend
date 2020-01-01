#ifdef __CUDACC__
#define CUDA_GLOBAL __global__
#define CUDA_HOST __host__
#define CUDA_DEVICE __device__
#define CUDA_DECLSPECS __host__ __device__
#else
#define CUDA_GLOBAL
#define CUDA_HOST
#define CUDA_DEVICE
#define CUDA_DECLSPECS
#endif
