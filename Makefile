CC=g++
CFLAGS=-Wall -Wextra -Wpedantic

all: tracer

vec3.o: src/vec3.cpp src/include/vec3.hpp
	$(CC) $(CFLAGS) -c src/vec3.cpp

tracer_main.o: src/tracer.cpp
	$(CC) $(CFLAGS) -c src/tracer.cpp

tracer: tracer_main.o vec3.o
	$(CC) $(CFLAGS) -o tracer tracer.o vec3.o

clean:
	@if test -n "$(wildcard *.o)"; then \
		rm *.o; \
		echo '.o files removed'; \
	fi
	@if test -n "$(wildcard tracer)"; then \
		rm tracer; \
		echo 'removed tracer'; \
	fi
