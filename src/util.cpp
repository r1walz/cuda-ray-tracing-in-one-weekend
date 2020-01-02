#include "include/util.hpp"

CUDA_GLOBAL
void initiate_world(int nx, int ny, float aperture, float dist_to_focus,
		    hittable **world, camera **cam
#ifdef __CUDACC__
		    , curandState *rand
#endif
		    ) {
#ifdef __CUDACC__
	curandState drand = *rand;
#endif
	int i = 0;
	hittable **list = new hittable*[22 * 22 + 4];
	list[i++] = new sphere(vec3(0.0f, -1000.0f, 0.0f), 1000.0f,
			new lambertian(vec3(0.5f, 0.5f, 0.5f)));

	for (int a = -11; a < 11; ++a)
	for (int b = -11; b < 11; ++b) {
		float choose =
#ifdef __CUDACC__
			       curand_uniform(&drand);
		vec3 center(a * curand_uniform(&drand), 0.2f, b * curand_uniform(&drand));
		if ((center - vec3(4.0f, 0.2f, 0.0f)).length() > 0.9f) {
			if (choose < 0.8f)
				list[i++] = new sphere(center, 0.2f,
						new lambertian(vec3(curand_uniform(&drand)
							       * curand_uniform(&drand),
							       curand_uniform(&drand)
							       * curand_uniform(&drand),
							       curand_uniform(&drand))
							       * curand_uniform(&drand)));
			else if (choose < 0.95f)
				list[i++] = new sphere(center, 0.2f,
						new metal(vec3(0.5f * (1.0f + curand_uniform(&drand)),
							       0.5f * (1.0f + curand_uniform(&drand)),
							       0.5f * (1.0f + curand_uniform(&drand))),
							  0.5f * curand_uniform(&drand)));
			else
				list[i++] = new sphere(center, 0.2f, new dielectric(1.5));
		}
#else
			       random_double();
		vec3 center(a * random_double(), 0.2f, b * random_double());
		if ((center - vec3(4.0f, 0.2f, 0.0f)).length() > 0.9f) {
			if (choose < 0.8f)
				list[i++] = new sphere(center, 0.2f,
						new lambertian(vec3(random_double()
							       * random_double(),
							       random_double()
							       * random_double(),
							       random_double())
							       * random_double()));
			else if (choose < 0.95f)
				list[i++] = new sphere(center, 0.2f,
						new metal(vec3(0.5f * (1.0f + random_double()),
							       0.5f * (1.0f + random_double()),
							       0.5f * (1.0f + random_double())),
							  0.5f * random_double()));
			else
				list[i++] = new sphere(center, 0.2f, new dielectric(1.5));
		}
#endif
	}

	list[i++] = new sphere(vec3(0.0f, 1.0f, 0.0f),
			       1.0f, new dielectric(1.5));
	list[i++] = new sphere(vec3(-4.0f, 1.0f, 0.0f),
			       1.0f, new lambertian(vec3(0.4f, 0.2f, 0.1f)));
	list[i++] = new sphere(vec3(4.0f, 1.0f, 0.0f),
			       1.0f, new metal(vec3(0.7f, 0.6f, 0.5f), 0.0f));

	*world = new hittable_list(list, i);
	*cam = new camera(vec3(13.0f, 2.0f, 3.0f),
			  vec3(0.0f, 0.0f, 0.0f),
			  vec3(0.0f, 1.0f, 0.0f),
			  20.0f, float(nx) / float(ny),
			  aperture, dist_to_focus);
}

CUDA_HOST
double random_double() {
	static std::uniform_real_distribution<double> distrib(0.0, 1.0);
	static std::mt19937 gen;
	static std::function<double()> rng = std::bind(distrib, gen);
	return rng();
}

CUDA_HOST
vec3 random_in_unit_disk() {
	vec3 p;

	do {
		p = 2.0f * vec3(random_double(), random_double(),
		    0.0f) - vec3(1.0f, 1.0f, 0.0f);
	} while (p.squared_length() >= 1.0f);

	return p;
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

CUDA_DEVICE
vec3 reflect(const vec3 &v, const vec3 &n) {
	return v - 2 * vec3::dot(v, n) * n;
}

CUDA_DEVICE
bool refract(const vec3 &v, const vec3 &n, float ni_over_nt, vec3 &refracted) {
	vec3 uv = vec3::unit_vector(v);
	float dt = vec3::dot(uv, n);
	float D = 1.0f - ni_over_nt * ni_over_nt * (1 - dt * dt);

	if (D > 0.0f) {
		refracted = ni_over_nt * (uv - n * dt) - n * sqrt(D);
		return true;
	}
	return false;
}

CUDA_DEVICE
float schlick(float cosine, float ref_idx) {
	float r0 = (1.0f - ref_idx) / (1.0f + ref_idx);
	r0 *= r0;

	return r0 + (1.0f - r0) * pow(1.0f - cosine, 5);
}

#ifdef __CUDACC__
CUDA_DEVICE
vec3 color(const ray& r, hittable *world, curandState *rand) {
	ray _ray = r;
	vec3 _attenuation = vec3(1.0f, 1.0f, 1.0f);

	for (int i = 0; i < 50; ++i) {
		hit_record rec;
		if (world->hit(_ray, 0.001f, MAXFLOAT, rec)) {
			ray scattered;
			vec3 attenuation;

			if (rec.mat_ptr->scatter(_ray, rec, attenuation,
						 scattered, rand)) {
				_attenuation *= attenuation;
				_ray = scattered;

			} else
				return vec3();
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
vec3 random_in_unit_disk(curandState *rand) {
	vec3 p;

	do {
		p = 2.0f * vec3(curand_uniform(rand), curand_uniform(rand),
		    0.0f) - vec3(1.0f, 1.0f, 0.0f);
	} while (p.squared_length() >= 1.0f);

	return p;
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
void init_random_item(curandState *rand) {
	curand_init(1999, 0, 0, rand);
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
		ray r((*cam)->get_ray(u, v, &drand));
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
	vec3 _attenuation = vec3(1.0f, 1.0f, 1.0f);

	for (int i = 0; i < 50; ++i) {
		hit_record rec;
		if (world->hit(_ray, 0.001f, MAXFLOAT, rec)) {
			ray scattered;
			vec3 attenuation;

			if (rec.mat_ptr->scatter(_ray, rec, attenuation,
						 scattered)) {
				_attenuation *= attenuation;
				_ray = scattered;

			} else
				return vec3();
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
