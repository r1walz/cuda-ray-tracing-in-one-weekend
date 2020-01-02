#ifndef _HITTABLELIST_H
#define _HITTABLELIST_H

#include "hittable.hpp"

class hittable_list : public hittable {
public:
	CUDA_DEVICE hittable_list() {}
	CUDA_DEVICE hittable_list(hittable **l, int n) : list(l), list_size(n) {}
	CUDA_DEVICE
	virtual bool hit(const ray &r,
			 float t_min,
			 float t_max,
			 hit_record &rec) const;
	hittable **list;
	int list_size;
};

#endif
