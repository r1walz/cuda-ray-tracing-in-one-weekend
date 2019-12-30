CC=g++
CFLAGS=-Wall -Wextra -Wpedantic

all: tracer

camera.o: src/camera.cpp src/include/camera.hpp
	$(CC) $(CFLAGS) -c src/camera.cpp

vec3.o: src/vec3.cpp src/include/vec3.hpp
	$(CC) $(CFLAGS) -c src/vec3.cpp

hitablelist.o: src/hitablelist.cpp src/include/hitablelist.hpp
	$(CC) $(CFLAGS) -c src/hitablelist.cpp

sphere.o: src/sphere.cpp src/include/sphere.hpp
	$(CC) $(CFLAGS) -c src/sphere.cpp

tracer_main.o: src/tracer.cpp
	$(CC) $(CFLAGS) -c src/tracer.cpp

lambertian.o: src/include/material.hpp src/lambertian.cpp
	$(CC) $(CFLAGS) -c src/lambertian.cpp

metal.o: src/include/material.hpp src/metal.cpp
	$(CC) $(CFLAGS) -c src/metal.cpp

dielectric.o: src/include/material.hpp src/dielectric.cpp
	$(CC) $(CFLAGS) -c src/dielectric.cpp

materials: lambertian.o metal.o dielectric.o

tracer: tracer_main.o camera.o vec3.o hitablelist.o sphere.o materials
	$(CC) $(CFLAGS) -o tracer tracer.o camera.o vec3.o sphere.o \
	hitablelist.o lambertian.o metal.o dielectric.o

clean:
	@if test -n "$(wildcard *.o)"; then \
		rm *.o; \
		echo '.o files removed'; \
	fi
	@if test -n "$(wildcard tracer)"; then \
		rm tracer; \
		echo 'removed tracer'; \
	fi
