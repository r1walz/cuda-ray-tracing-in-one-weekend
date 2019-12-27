#ifndef _VEC3_H
#define _VEC3_H

#include <math.h>
#include <stdlib.h>
#include <iostream>

class vec3 {
public:
	vec3() : e{ 0.0f, 0.0f, 0.0f } {}
	vec3(float e0, float e1, float e2) : e{ e0, e1, e2 } {}

	inline float x() const { return e[0]; }
	inline float y() const { return e[1]; }
	inline float z() const { return e[2]; }
	inline float r() const { return e[0]; }
	inline float g() const { return e[1]; }
	inline float b() const { return e[2]; }

	static vec3 unit_vector(const vec3&);
	static float dot(const vec3&, const vec3&);
	static vec3 cross(const vec3&, const vec3&);

	void make_unit_vector();
	inline float length() const { return sqrt(squared_length()); }
	inline float squared_length() const { return e[0] * e[0] + e[1] * e[1] + e[2] * e[2]; }

	inline const vec3& operator+() const { return *this; }
	inline float operator[](int i) const { return e[i]; }
	inline float& operator[](int i) { return e[i]; }
	inline vec3 operator-() const { return vec3(-e[0], -e[1], -e[2]); }

	vec3& operator+=(const vec3&);
	vec3& operator-=(const vec3&);
	vec3& operator*=(const vec3&);
	vec3& operator/=(const vec3&);
	vec3& operator*=(const float);
	vec3& operator/=(const float);

	friend std::istream& operator>>(std::istream&, vec3&);
	friend std::ostream& operator<<(std::ostream&, const vec3&);
	friend vec3 operator+(const vec3&, const vec3&);
	friend vec3 operator-(const vec3&, const vec3&);
	friend vec3 operator*(const vec3&, const vec3&);
	friend vec3 operator/(const vec3&, const vec3&);
	friend vec3 operator*(float, const vec3&);
	friend vec3 operator/(const vec3&, float);
	friend vec3 operator*(const vec3&, float);

	float e[3];
};

#endif
