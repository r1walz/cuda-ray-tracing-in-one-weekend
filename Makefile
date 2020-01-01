CC=g++
CFLAGS=-Wall -Wextra -Wpedantic

ifeq ($(CC), nvcc)
CFLAGS=-x cu
endif

all: tracer

util.o: src/util.cpp src/include/util.hpp
	$(CC) $(CFLAGS) -c src/util.cpp

vec3.o: src/vec3.cpp src/include/vec3.hpp
	$(CC) $(CFLAGS) -c src/vec3.cpp

tracer.o: src/tracer.cpp
	$(CC) $(CFLAGS) -c src/tracer.cpp

tracer: tracer.o util.o vec3.o
	$(CC) -o tracer tracer.o util.o vec3.o

clean:
	@if test -n "$(wildcard *.o)"; then \
		rm *.o; \
		echo 'removed .o files'; \
	fi
	@if test -n "$(wildcard tracer)"; then \
		rm tracer; \
		echo 'removed tracer'; \
	fi
