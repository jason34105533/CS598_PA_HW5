# Campus Cluster build.  Load the toolchain first:
#     source env.sh
#
# The eng-instruction AMD nodes are EPYC 7713 (Zen 3) while the login node is
# Genoa (Zen 4), so -march=native would emit AVX-512 that the compute nodes
# cannot run.  Target Zen 3 explicitly.

CC     = mpicc
CFLAGS = -O3 -march=znver3 -std=gnu11 -fopenmp-simd -Wall -Wextra
LDLIBS = -lm

SRC = src/main.c src/grid.c src/jacobi.c src/pcg.c src/coarse.c src/refsol.c
OBJ = $(SRC:.c=.o)

poisson: $(OBJ)
	$(CC) $(CFLAGS) $(OBJ) -o $@ $(LDLIBS)

src/%.o: src/%.c src/grid.h src/solver.h src/coarse.h
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f poisson src/*.o tests/test_coarse

.PHONY: clean

tests/test_coarse: tests/test_coarse.c src/coarse.o src/grid.o
	$(CC) $(CFLAGS) $^ -o $@ $(LDLIBS)
