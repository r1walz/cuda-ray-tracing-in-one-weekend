#include "include/camera.hpp"

CUDA_DEVICE
ray camera::get_ray(float u, float v) {
	return ray(origin, lower_left_corner
			   + u * horizontal + v * vertical - origin);
}
