CC=g++
CFLAGS=-Wall -Wextra -Wpedantic

all: main

main: src/main.cpp
	$(CC) $(CFLAGS) src/main.cpp -o main

clean:
	@if test -n "$(wildcard main)"; then \
		rm main; \
		echo 'removed main'; \
	fi
