#include "include/camera.hpp"

CUDA_DEVICE
camera::camera(vec3 lookfrom, vec3 lookat, vec3 vup,
	       float vfov, float aspect, float aperture, float focus_dist) :
	origin(lookfrom), lens_radius(aperture / 2)
{
	w = vec3::unit_vector(lookfrom - lookat);
	u = vec3::unit_vector(vec3::cross(vup, w));
	v = vec3::cross(w, u);
	float theta = vfov * M_PI / 180;
	float half_height = tan(theta / 2);
	float half_width = aspect * half_height;
	lower_left_corner = origin - focus_dist * (half_width * u
						   + half_height * v + w);
	horizontal = 2 * half_width * focus_dist * u;
	vertical = 2 * half_height * focus_dist * v;
}

CUDA_DEVICE
ray camera::get_ray(float s, float t
#ifdef __CUDACC__
		    , curandState *rand
		    ) {
	vec3 rd = lens_radius * random_in_unit_disk(rand);
#else
		    ) {
	vec3 rd = lens_radius * random_in_unit_disk();
#endif
	vec3 offset = u * rd.x() + v * rd.y();

	return ray(origin + offset, lower_left_corner
				    + s * horizontal
				    + t * vertical
				    - origin - offset);
}
