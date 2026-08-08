# Host-side build for the vio_hold test harness.
# The library itself is plain C11 with no OS or heap dependencies; on the
# STM32N6 just compile src/*.c with your arm-none-eabi toolchain
# (-mcpu=cortex-m55 -mfloat-abi=hard -O2 ...) and add include/ to the path.

CC      ?= cc
CFLAGS  ?= -std=c11 -O2 -Wall -Wextra -Wconversion -Iinclude
LDLIBS   = -lm

SRC  = src/vh_fast.c src/vh_pyramid.c src/vh_klt.c src/vh_rotcomp.c src/vh_hold.c src/vh_bias.c
OBJ  = $(SRC:.c=.o)

build/test_pipeline: tests/test_pipeline.c $(SRC)
	@mkdir -p build
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)

.PHONY: test clean
test: build/test_pipeline
	./build/test_pipeline

clean:
	rm -rf build $(OBJ)
