#include "include/util.hpp"

CUDA_GLOBAL
void initiate_world(hittable **list, hittable **world, camera **cam) {
	list[0] = new sphere(vec3(0.0f, 0.0f, -1.0f), 0.5f);
	list[1] = new sphere(vec3(0.0f, -100.5f, -1.0f), 100.0f);
	*world = new hittable_list(list, 2);
	*cam = new camera();
}

CUDA_HOST
double random_double() {
	static std::uniform_real_distribution<double> distrib(0.0, 1.0);
	static std::mt19937 gen;
	static std::function<double()> rng = std::bind(distrib, gen);
	return rng();
}

CUDA_HOST
vec3 random_in_unit_sphere() {
	vec3 p;

	do {
		p = 2.0f * vec3(random_double(), random_double(),
				random_double()) - vec3(1.0f, 1.0f, 1.0f);
	} while (p.squared_length() >= 1.0f);

	return p;
}

#ifdef __CUDACC__
CUDA_DEVICE
vec3 color(const ray& r, hittable *world, curandState *rand) {
	ray _ray = r;
	float _attenuation = 1.0f;

	for (int i = 0; i < 50; ++i) {
		hit_record rec;
		if (world->hit(_ray, 0.001f, MAXFLOAT, rec)) {
			vec3 target = rec.p + rec.normal + random_in_unit_sphere(rand);
			_attenuation *= 0.5f;
			_ray = ray(rec.p, target - rec.p);
		} else {
			vec3 unit_direction = vec3::unit_vector(_ray.direction());
			float t = 0.5f * (unit_direction.y() + 1.0f);
			return _attenuation * ((1.0f - t) * vec3(1.0f, 1.0f, 1.0f)
			       + t * vec3(0.5f, 0.7f, 1.0f));
		}
	}
	return vec3();
}

CUDA_DEVICE
vec3 random_in_unit_sphere(curandState *rand) {
	vec3 p;

	do {
		p = 2.0f * vec3(curand_uniform(rand), curand_uniform(rand),
				curand_uniform(rand)) - vec3(1.0f, 1.0f, 1.0f);
	} while (p.squared_length() >= 1.0f);

	return p;
}

CUDA_GLOBAL
void init_random(int nx, int ny, curandState *rand) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (int i = idx; i < nx * ny; i += stride)
		curand_init(1999 + i, 0, 0, &rand[i]);
}

CUDA_GLOBAL
void paint_pixel(int nx, int ny, int ns, camera **cam,
		 hittable **world, curandState *rand, float *output) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (int i = idx; i < nx * ny * ns; i += stride) {
		curandState drand = rand[i % (nx * ny)];
		float u = float(i + curand_uniform(&drand)) / float(nx * ny * ns);
		float v = float((i % (ny * ns)) + curand_uniform(&drand)) / float(ny * ns);
		ray r((*cam)->get_ray(u, v));
		vec3 col = color(r, *world, &drand);
		output[i * 3 ] = col[0];
		output[i * 3 + 1] = col[1];
		output[i * 3 + 2] = col[2];
	}
}
#else
CUDA_HOST
vec3 color(const ray& r, hittable *world) {
	ray _ray = r;
	float _attenuation = 1.0f;

	for (int i = 0; i < 50; ++i) {
		hit_record rec;
		if (world->hit(_ray, 0.001f, MAXFLOAT, rec)) {
			vec3 target = rec.p + rec.normal + random_in_unit_sphere();
			_attenuation *= 0.5f;
			_ray = ray(rec.p, target - rec.p);
		} else {
			vec3 unit_direction = vec3::unit_vector(_ray.direction());
			float t = 0.5f * (unit_direction.y() + 1.0f);
			return _attenuation * ((1.0f - t) * vec3(1.0f, 1.0f, 1.0f)
			       + t * vec3(0.5f, 0.7f, 1.0f));
		}
	}
	return vec3();
}

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
#endif
