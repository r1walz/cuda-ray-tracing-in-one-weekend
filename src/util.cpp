#include "include/util.hpp"

CUDA_GLOBAL
void initiate_world(hittable **list, hittable **world) {
	list[0] = new sphere(vec3(0.0f, 0.0f, -1.0f), 0.5f);
	list[1] = new sphere(vec3(0.0f, -100.5f, -1.0f), 100.0f);
	*world = new hittable_list(list, 2);
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

CUDA_GLOBAL
void paint_pixel(int nx, int ny, const vec3 *origin, const vec3 *vertical,
		 const vec3 *horizontal, const vec3 *lower_left_corner,
		 hittable **world, float *output) {
#ifdef __CUDACC__
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = blockDim.x * gridDim.x;

	for (int i = idx; i < nx * ny; i += stride) {
#else
	for (int i = 0; i < nx * ny; ++i) {
#endif
		float u = float(i / ny) / float(nx);
		float v = float(i % ny) / float(ny);
		ray r(*origin, *lower_left_corner
			       + u * *horizontal + v * *vertical);
		vec3 col = color(r, *world);
		output[i * 3] = col[0];
		output[i * 3 + 1] = col[1];
		output[i * 3 + 2] = col[2];
	}
}
