#ifndef _UTIL_H
#define _UTIL_H

#include "directives.hpp"

CUDA_GLOBAL void paint_pixel(int nx, int ny, float *output);

#endif
