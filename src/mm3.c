#include <stdio.h>
#include <stdlib.h>
#include <immintrin.h>  // Include for AVX

// Forward declaration of the matrix multiplication function
void mm(const int M, const int N, const int K, const float *a, const float *b, float *c);

// Inner kernel of matrix multiplication using AVX
void matmul_dot_inner(const float *a, const float *b, float *c, const int M, const int N, const int K, const int m, const int n) {
    __m256 csum[3][4] = {{ _mm256_setzero_ps() }};
    for (int k = 0; k < K; k++) {
        for (unsigned ai = 0; ai < 3; ai++) {
            __m256 aa = _mm256_set1_ps(a[(m + ai) * K + k]);
            for (unsigned bi = 0; bi < 4; bi++) {
                __m256 bb = _mm256_loadu_ps(&b[k * N + n + bi * 8]);
                csum[ai][bi] = _mm256_fmadd_ps(aa, bb, csum[ai][bi]);
            }
        }
    }

    for (unsigned ai = 0; ai < 3; ai++) {
        for (unsigned bi = 0; bi < 4; bi++) {
            _mm256_storeu_ps(&c[(m + ai) * N + n + bi * 8], csum[ai][bi]);
        }
    }
}

void mm(const int M, const int N, const int K, const float *a, const float *b, float *c) {
    for (int m = 0; m < M; m += 3) {
        for (int n = 0; n < N; n += 4) {
            matmul_dot_inner(a, b, c, M, N, K, m, n);
        }
    }
}

int main() {
    const int M = 960; // Rows in A and C
    const int N = 960; // Columns in B and C
    const int K = 960; // Columns in A and rows in B

    // Allocate memory for matrices A, B, and C
    float *a = (float *) malloc(M * K * sizeof(float));
    float *b = (float *) malloc(K * N * sizeof(float));
    float *c = (float *) malloc(M * N * sizeof(float));

    // Initialize matrices A and B with some values for testing
    for (int i = 0; i < M * K; i++) {
        a[i] = i % 100 * 0.01;
    }
    for (int i = 0; i < K * N; i++) {
        b[i] = i % 100 * 0.01;
    }

    // Set matrix C to zero
    for (int i = 0; i < M * N; i++) {
        c[i] = 0.0;
    }

    // Perform matrix multiplication
    mm(M, N, K, a, b, c);

    // Optionally print some elements of matrix C to verify results
    for (int i = 0; i < 10; i++) {
        printf("%f ", c[i]);
    }
    printf("\n");

    // Free allocated memory
    free(a);
    free(b);
    free(c);

    return 0;
}

