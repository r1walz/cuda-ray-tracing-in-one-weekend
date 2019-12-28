#include "include/hitablelist.hpp"

bool hitable_list::hit(const ray &r,
		       float t_min,
		       float t_max,
		       hit_record &rec) const {
	hit_record tmp;
	bool hit_anything = false;
	double closest_so_far = t_max;

	for (int i = 0; i < list_size; ++i)
		if (list[i]->hit(r, t_min, closest_so_far, tmp)) {
			hit_anything = true;
			closest_so_far = tmp.t;
			rec = tmp;
		}
	return hit_anything;
}
