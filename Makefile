CC=g++
CFLAGS=-Wall -Wextra -Wpedantic

ifeq ($(CC), nvcc)
CFLAGS=-x cu
endif

all: tracer

util.o: src/util.cpp src/include/util.hpp
	$(CC) $(CFLAGS) -c src/util.cpp

tracer.o: src/tracer.cpp
	$(CC) $(CFLAGS) -c src/tracer.cpp

tracer: tracer.o util.o
	$(CC) -o tracer tracer.o util.o

clean:
	@if test -n "$(wildcard *.o)"; then \
		rm *.o; \
		echo 'removed .o files'; \
	fi
	@if test -n "$(wildcard tracer)"; then \
		rm tracer; \
		echo 'removed tracer'; \
	fi
