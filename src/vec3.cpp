#include "include/vec3.hpp"

CUDA_DECLSPECS
vec3 vec3::unit_vector(const vec3 &v) {
	return v / v.length();
}

CUDA_DECLSPECS
void vec3::make_unit_vector() {
	float k = 1.0 / length();
	e[0] *= k;
	e[1] *= k;
	e[2] *= k;
}

CUDA_DECLSPECS
float vec3::dot(const vec3 &v1, const vec3 &v2) {
	return v1.e[0] * v2.e[0] +
	       v1.e[1] * v2.e[1] +
	       v1.e[2] * v2.e[2];
}

CUDA_DECLSPECS
vec3 vec3::cross(const vec3 &v1, const vec3 &v2) {
	return vec3(v1.e[1] * v2.e[2] - v1.e[2] * v2.e[1],
		    v1.e[2] * v2.e[0] - v1.e[0] * v2.e[2],
		    v1.e[0] * v2.e[1] - v1.e[1] * v2.e[0]);
}

CUDA_DECLSPECS
vec3& vec3::operator+=(const vec3 &v) {
	e[0] += v.e[0];
	e[1] += v.e[1];
	e[2] += v.e[2];
	return *this;
}

CUDA_DECLSPECS
vec3& vec3::operator*=(const vec3 &v) {
	e[0] *= v.e[0];
	e[1] *= v.e[1];
	e[2] *= v.e[2];
	return *this;
}

CUDA_DECLSPECS
vec3& vec3::operator/=(const vec3 &v) {
	e[0] /= v.e[0];
	e[1] /= v.e[1];
	e[2] /= v.e[2];
	return *this;
}

CUDA_DECLSPECS
vec3& vec3::operator-=(const vec3 &v) {
	e[0] -= v.e[0];
	e[1] -= v.e[1];
	e[2] -= v.e[2];
	return *this;
}

CUDA_DECLSPECS
vec3& vec3::operator*=(const float t) {
	e[0] *= t;
	e[1] *= t;
	e[2] *= t;
	return *this;
}

CUDA_DECLSPECS
vec3& vec3::operator/=(const float t) {
	float k = 1.0 / t;
	e[0] *= k;
	e[1] *= k;
	e[2] *= k;
	return *this;
}

CUDA_HOST
std::istream& operator>>(std::istream &is, vec3 &t) {
	is >> t.e[0] >> t.e[1] >> t.e[2];
	return is;
}

CUDA_HOST
std::ostream& operator<<(std::ostream &os, const vec3 &t) {
	os << t.e[0] << " " << t.e[1] << " " << t.e[2];
	return os;
}

CUDA_DECLSPECS
vec3 operator+(const vec3 &v1, const vec3 &v2) {
	return vec3(v1.e[0] + v2.e[0],
		    v1.e[1] + v2.e[1],
		    v1.e[2] + v2.e[2]);
}

CUDA_DECLSPECS
vec3 operator-(const vec3 &v1, const vec3 &v2) {
	return vec3(v1.e[0] - v2.e[0],
		    v1.e[1] - v2.e[1],
		    v1.e[2] - v2.e[2]);
}


CUDA_DECLSPECS
vec3 operator*(const vec3 &v1, const vec3 &v2) {
	return vec3(v1.e[0] * v2.e[0],
		    v1.e[1] * v2.e[1],
		    v1.e[2] * v2.e[2]);
}

CUDA_DECLSPECS
vec3 operator/(const vec3 &v1, const vec3 &v2) {
	return vec3(v1.e[0] / v2.e[0],
		    v1.e[1] / v2.e[1],
		    v1.e[2] / v2.e[2]);
}

CUDA_DECLSPECS
vec3 operator*(float t, const vec3 &v) {
	return vec3(t * v.e[0], t * v.e[1], t * v.e[2]);
}

CUDA_DECLSPECS
vec3 operator*(const vec3 &v, float t) {
	return vec3(t * v.e[0], t * v.e[1], t * v.e[2]);
}

CUDA_DECLSPECS
vec3 operator/(const vec3 &v, float t) {
	return vec3(v.e[0] / t, v.e[1] / t, v.e[2] / t);
}
