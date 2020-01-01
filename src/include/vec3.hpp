#ifndef _VEC3_H
#define _VEC3_H

#include <cmath>
#include <iostream>
#include "directives.hpp"

class vec3 {
public:
	CUDA_DECLSPECS vec3() : e{ 0.0f, 0.0f, 0.0f } {}
	CUDA_DECLSPECS vec3(float e0, float e1, float e2) : e{ e0, e1, e2 } {}

	CUDA_DECLSPECS inline float x() const { return e[0]; }
	CUDA_DECLSPECS inline float y() const { return e[1]; }
	CUDA_DECLSPECS inline float z() const { return e[2]; }
	CUDA_DECLSPECS inline float r() const { return e[0]; }
	CUDA_DECLSPECS inline float g() const { return e[1]; }
	CUDA_DECLSPECS inline float b() const { return e[2]; }

	CUDA_DECLSPECS static vec3 unit_vector(const vec3&);
	CUDA_DECLSPECS static float dot(const vec3&, const vec3&);
	CUDA_DECLSPECS static vec3 cross(const vec3&, const vec3&);

	CUDA_DECLSPECS void make_unit_vector();
	CUDA_DECLSPECS inline float length() const { return sqrt(squared_length()); }
	CUDA_DECLSPECS inline float squared_length() const { return e[0] * e[0] + e[1] * e[1] + e[2] * e[2]; }

	CUDA_DECLSPECS inline const vec3& operator+() const { return *this; }
	CUDA_DECLSPECS inline float operator[](int i) const { return e[i]; }
	CUDA_DECLSPECS inline float& operator[](int i) { return e[i]; }
	CUDA_DECLSPECS inline vec3 operator-() const { return vec3(-e[0], -e[1], -e[2]); }

	CUDA_DECLSPECS vec3& operator+=(const vec3&);
	CUDA_DECLSPECS vec3& operator-=(const vec3&);
	CUDA_DECLSPECS vec3& operator*=(const vec3&);
	CUDA_DECLSPECS vec3& operator/=(const vec3&);
	CUDA_DECLSPECS vec3& operator*=(const float);
	CUDA_DECLSPECS vec3& operator/=(const float);

	CUDA_HOST friend std::istream& operator>>(std::istream&, vec3&);
	CUDA_HOST friend std::ostream& operator<<(std::ostream&, const vec3&);
	CUDA_DECLSPECS friend vec3 operator+(const vec3&, const vec3&);
	CUDA_DECLSPECS friend vec3 operator-(const vec3&, const vec3&);
	CUDA_DECLSPECS friend vec3 operator*(const vec3&, const vec3&);
	CUDA_DECLSPECS friend vec3 operator/(const vec3&, const vec3&);
	CUDA_DECLSPECS friend vec3 operator*(float, const vec3&);
	CUDA_DECLSPECS friend vec3 operator/(const vec3&, float);
	CUDA_DECLSPECS friend vec3 operator*(const vec3&, float);

	float e[3];
};

#endif
