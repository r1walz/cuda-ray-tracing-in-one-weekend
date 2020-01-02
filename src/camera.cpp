#include "include/camera.hpp"

CUDA_DEVICE
camera::camera(vec3 lookfrom, vec3 lookat, vec3 vup, float vfov, float aspect) :
	origin(lookfrom)
{
	vec3 w(vec3::unit_vector(lookfrom - lookat));
	vec3 u(vec3::unit_vector(vec3::cross(vup, w)));
	vec3 v(vec3::cross(w, u));
	float theta = vfov * M_PI / 180;
	float half_height = tan(theta / 2);
	float half_width = aspect * half_height;
	lower_left_corner = origin - half_width * u - half_height * v - w;
	horizontal = 2 * half_width * u;
	vertical = 2 * half_height * v;
}

CUDA_DEVICE
ray camera::get_ray(float u, float v) {
	return ray(origin, lower_left_corner
			   + u * horizontal + v * vertical - origin);
}
