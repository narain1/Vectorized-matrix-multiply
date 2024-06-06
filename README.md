# Optimized Matrix Multiplication

This repository showcases various implementations of matrix multiplication, each demonstrating significant performance improvements through different optimization techniques. These methods include vectorized computations using AVX2 intrinsics, loop tiling to enhance cache efficiency, and parallel processing with OpenMP. The goal is to highlight how these optimizations can dramatically reduce computation times in large-scale matrix multiplication tasks.

## Repository Structure

The repository contains multiple C programs, each tailored to demonstrate a specific optimization technique:

- **AVX2 Optimized Matrix Multiplication:** Utilizes AVX2 intrinsics to perform efficient vectorized multiplication of floating-point numbers, significantly speeding up operations by leveraging SIMD (Single Instruction Multiple Data) capabilities.
- **Basic Matrix Multiplication with Loop Tiling:** Implements loop tiling (also known as loop blocking) to enhance cache utilization, which helps to minimize memory access latency and improve execution speed.
- **Parallel Matrix Multiplication with OpenMP:** Employs OpenMP to parallelize computation across multiple CPU cores, effectively distributing the workload to accelerate processing.
- **Reduced Vector Operations:** Features methods for efficiently summing elements within an AVX2 vector, optimizing the computational throughput of vectorized operations.

## Compilation

Each program should be compiled using `gcc` or a compatible C compiler. It's essential to activate compiler optimizations and, where relevant, specify the target architecture to incorporate AVX2 instructions, ensuring the best possible performance.

### Example Compilation Command

```bash
gcc -O3 -mavx2 -fopenmp program_name.c -o program_name
```

This command compiles a program enabling Level 3 optimizations (`-O3`), AVX2 instructions (`-mavx2`), and OpenMP support (`-fopenmp`).

## Dependencies

- **C Compiler:** gcc or another compiler supporting C99 or later.
- **AVX2 Support:** Hardware that supports AVX2 instructions is required for certain programs to run.
- **OpenMP:** Necessary for the execution of parallelized programs.

## Performance Measurement

Execution times for each program are measured using high-resolution clocks, and results are output to the standard output (stdout) in seconds. This setup allows for straightforward comparisons between different optimization techniques, providing clear insights into the efficiency gains possible with each approach.

