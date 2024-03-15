# Vectorized-matrix-multiply

## naive implementation

o0 optimization 168.695055 seconds
o1 optimization 88.203721 seconds
o2 optimization 21.618243 seconds
o3 optimization 22.122448 seconds

# Optimized Matrix Multiplication

This repository contains various implementations of matrix multiplication, showcasing the performance improvements achievable through different optimization techniques. These optimizations include the use of AVX2 intrinsics, loop tiling, and parallel processing with OpenMP. The aim is to demonstrate how these methods can significantly reduce computation time for large-scale matrix multiplication tasks.

## Repository Structure

The repository includes multiple C programs, each demonstrating a unique optimization strategy:

- **AVX2 Optimized Matrix Multiplication**: Utilizes AVX2 intrinsics to perform efficient vectorized multiplication of floating-point numbers.
- **Basic Matrix Multiplication with Loop Tiling**: Implements loop tiling (also known as loop blocking) to improve cache utilization and reduce memory access latency.
- **Parallel Matrix Multiplication with OpenMP**: Leverages OpenMP to parallelize the computation, distributing the workload across multiple CPU cores.
- **Reduced Vector Operations**: Showcases two methods for summing elements within an AVX2 vector, further optimizing the vectorized approach.

## Compilation

Each program can be compiled with `gcc` or any compatible C compiler. It is crucial to enable optimizations and, where applicable, specify the target architecture to support AVX2 instructions. An example compilation command for an AVX2-enabled program is:

```bash
gcc -O3 -mavx2 -fopenmp program_name.c -o program_name
```

## Dependencies

* A C compiler such as gcc supporting C99 or later.
* Hardware support for AVX2 instructions for certain programs.
* OpenMP for parallel execution in applicable programs.

## Performance Measurement

Each program measures execution time using high-resolution clocks. The results are printed to stdout in seconds, allowing for easy comparison between the different optimization techniques.

