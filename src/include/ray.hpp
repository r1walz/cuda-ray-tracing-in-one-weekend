#ifndef _RAY_H
#define _RAY_H

#include "vec3.hpp"
#include "directives.hpp"

class ray {
public:
	CUDA_DEVICE ray() {}
	CUDA_DEVICE ray(const vec3& a, const vec3 &b) : A(a), B(b) {}

	CUDA_DEVICE vec3 origin() const { return A; }
	CUDA_DEVICE vec3 direction() const { return B; }
	CUDA_DEVICE vec3 point_at_parameter(float t) const { return A + t * B; }

	vec3 A, B;
};

#endif
