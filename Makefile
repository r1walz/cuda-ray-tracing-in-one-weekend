CC=g++
CFLAGS=-Wall -Wextra -Wpedantic -c

ifeq ($(CC), nvcc)
CFLAGS=-x cu -dc
endif

all: tracer

util.o: src/util.cpp src/include/util.hpp
	$(CC) $(CFLAGS) src/util.cpp

vec3.o: src/vec3.cpp src/include/vec3.hpp
	$(CC) $(CFLAGS) src/vec3.cpp

camera.o: src/camera.cpp src/include/camera.hpp
	$(CC) $(CFLAGS) src/camera.cpp

hittable_list.o: src/hittable_list.cpp src/include/hittable_list.hpp
	$(CC) $(CFLAGS) src/hittable_list.cpp

sphere.o: src/sphere.cpp src/include/sphere.hpp
	$(CC) $(CFLAGS) src/sphere.cpp

hittables: hittable_list.o sphere.o

lambertian.o: src/lambertian.cpp
	$(CC) $(CFLAGS) src/lambertian.cpp

metal.o: src/metal.cpp
	$(CC) $(CFLAGS) src/metal.cpp

dielectric.o: src/dielectric.cpp
	$(CC) $(CFLAGS) src/dielectric.cpp

materials: lambertian.o metal.o dielectric.o

tracer.o: src/tracer.cpp
	$(CC) $(CFLAGS) src/tracer.cpp

tracer: tracer.o util.o vec3.o camera.o hittables materials
	$(CC) -o tracer tracer.o util.o vec3.o camera.o hittable_list.o \
	sphere.o lambertian.o metal.o dielectric.o

clean:
	@if test -n "$(wildcard *.o)"; then \
		rm *.o; \
		echo 'removed .o files'; \
	fi
	@if test -n "$(wildcard tracer)"; then \
		rm tracer; \
		echo 'removed tracer'; \
	fi
