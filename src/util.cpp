#include "include/util.hpp"

CUDA_GLOBAL
void initiate_world(hittable **list, hittable **world, camera **cam) {
	list[0] = new sphere(vec3(0.0f, 0.0f, -1.0f), 0.5f);
	list[1] = new sphere(vec3(0.0f, -100.5f, -1.0f), 100.0f);
	*world = new hittable_list(list, 2);
	*cam = new camera();
}

CUDA_DEVICE
vec3 color(const ray& r, hittable *world) {
	hit_record rec;
	if (world->hit(r, 0.0f, MAXFLOAT, rec))
		return 0.5 * vec3(rec.normal.x() + 1.0f,
				  rec.normal.y() + 1.0f,
				  rec.normal.z() + 1.0f);
	vec3 unit_direction = vec3::unit_vector(r.direction());
	float t = 0.5f * (unit_direction.y() + 1.0f);

	return (1.0f - t) * vec3(1.0f, 1.0f, 1.0f) + t * vec3(0.5f, 0.7f, 1.0f);
}

CUDA_HOST
double random_double() {
	static std::uniform_real_distribution<double> distrib(0.0, 1.0);
	static std::mt19937 gen;
	static std::function<double()> rng = std::bind(distrib, gen);
	return rng();
}

#ifdef __CUDACC__
CUDA_GLOBAL
void init_random(int nx, int ny, curandState *rand) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (int i = idx; i < nx * ny; i += stride)
		curand_init(1999 + i, 0, 0, &rand[i]);
}
#endif

CUDA_GLOBAL
void paint_pixel(int nx, int ny, int ns, camera **cam,
		 hittable **world, float *output) {
	for (int i = 0; i < nx * ny * ns; ++i) {
		float u = float(i) / float(nx * ny * ns);
		float v = float((i % (ny * ns))) / float(ny * ns);
		ray r((*cam)->get_ray(u, v));
		vec3 col = color(r, *world);
		output[i * 3 ] = col[0];
		output[i * 3 + 1] = col[1];
		output[i * 3 + 2] = col[2];
	}
}

#ifdef __CUDACC__
CUDA_GLOBAL
void paint_pixel(int nx, int ny, int ns, camera **cam,
		 hittable **world, curandState *rand, float *output) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (int i = idx; i < nx * ny * ns; i += stride) {
		curandState drand = rand[i / ns];
		float u = float(i + curand_uniform(&drand)) / float(nx * ny * ns);
		float v = float((i % (ny * ns)) + curand_uniform(&drand)) / float(ny * ns);
		ray r((*cam)->get_ray(u, v));
		vec3 col = color(r, *world);
		output[i * 3 ] = col[0];
		output[i * 3 + 1] = col[1];
		output[i * 3 + 2] = col[2];
	}
}
#endif
