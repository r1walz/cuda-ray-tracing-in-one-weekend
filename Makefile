CC=g++
CFLAGS=-Wall -Wextra -Wpedantic

all: tracer

vec3.o: src/vec3.cpp src/include/vec3.hpp
	$(CC) $(CFLAGS) -c src/vec3.cpp

hitablelist.o: src/hitablelist.cpp src/include/hitablelist.hpp
	$(CC) $(CFLAGS) -c src/hitablelist.cpp

sphere.o: src/sphere.cpp src/include/sphere.hpp
	$(CC) $(CFLAGS) -c src/sphere.cpp

tracer_main.o: src/tracer.cpp
	$(CC) $(CFLAGS) -c src/tracer.cpp

tracer: tracer_main.o vec3.o hitablelist.o sphere.o
	$(CC) $(CFLAGS) -o tracer tracer.o vec3.o sphere.o hitablelist.o

clean:
	@if test -n "$(wildcard *.o)"; then \
		rm *.o; \
		echo '.o files removed'; \
	fi
	@if test -n "$(wildcard tracer)"; then \
		rm tracer; \
		echo 'removed tracer'; \
	fi
