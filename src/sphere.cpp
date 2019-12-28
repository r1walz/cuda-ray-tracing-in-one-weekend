#include "include/sphere.hpp"

bool sphere::hit(const ray &r, float t_min, float t_max, hit_record &rec) const {
	vec3 oc = r.origin() - center;
	float a = vec3::dot(r.direction(), r.direction());
	float b = vec3::dot(oc, r.direction());
	float c = vec3::dot(oc, oc) - radius * radius;
	float D = b * b - a * c;

	if (D > 0.0f) {
		float tmp = (-b - sqrt(D)) / a;
		if (tmp > t_min && tmp < t_max) {
			rec.t = tmp;
			rec.p = r.point_at_parameter(rec.t);
			rec.normal = (rec.p - center) / radius;
			rec.mat_ptr = mat_ptr;
			return true;
		}
		tmp = (-b + sqrt(D)) / a;
		if (tmp > t_min && tmp < t_max) {
			rec.t = tmp;
			rec.p = r.point_at_parameter(rec.t);
			rec.normal = (rec.p - center) / radius;
			rec.mat_ptr = mat_ptr;
			return true;
		}
	}
	return false;
}
