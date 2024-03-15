#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <immintrin.h>

#define n 4096
#define sz 8


float reduce_vector8_0(__m256 v) {
  __m128 v1 = _mm256_extractf128_ps(v, 0);
  __m128 v2 = _mm256_extractf128_ps(v, 1);
  __m128 v3 = _mm_add_ps(v1, v2);
  __m128 v4 = _mm_shuffle_ps(v3, v3, 0x4e);
  __m128 v5 = _mm_add_ps(v3, v4);
  __m128 v6 = _mm_shuffle_ps(v5, v5, 0x11);
  __m128 v7 = _mm_add_ps(v5, v6);
  return _mm_cvtss_f32(v7);
}


float reduce_vector8_1(__m256 v) {
    __m256 sum = _mm256_hadd_ps(v, v); // Horizontal add within the 256-bit vector
    sum = _mm256_hadd_ps(sum, sum); // Further horizontal add to accumulate the sum
    __m128 sum128 = _mm256_extractf128_ps(sum, 0); // Extract lower 128 bits
    sum128 = _mm_add_ps(sum128, _mm256_extractf128_ps(sum, 1)); // Add upper 128 bits to lower 128 bits
    return _mm_cvtss_f32(_mm_hadd_ps(sum128, sum128)); // Horizontal add within the 128-bit vector and return the result
}


void matrix_multiply_avx2(float *A, float *B, float *C) {
  const int ncols8 = n & ~7;

  for (int i =0; i<n; i += sz) {
    for (int j=0; j<n; j += sz) {
      for (int k=0; k<n; k += sz) {
        for (int ii=i; ii < i+sz; ii++) {
          for (int jj=j; jj< j+sz; jj++) {
            __m256 sum = _mm256_setzero_ps();
            for (int kk=k; kk < k+sz; k+=8) {
              __m256 a = _mm256_loadu_ps(A + ii * n + kk);
              __m256 b = _mm256_loadu_ps(B + jj * n + kk);
              __m256 c = _mm256_mul_ps(a, b);
              sum = _mm256_add_ps(sum, c);
            }
            C[i] = reduce_vector8_1(sum);

            for (int j=ncols8; j<n; j++) {
            C[i] += A[i * n + j]* B[j];
            }
          }
        }
      }
    }
  }
}


int main() {
  time_t t;
  float *a = (float *)malloc(sizeof(float)*n*n);
  float *b = (float *)malloc(sizeof(float)*n*n);
  float *c = (float *)malloc(sizeof(float)*n*n);

  srand((unsigned) time(&t));
  for (int i=0; i<n;i++) 
    for (int j=0; j<n; j++) {
      a[i] = (float)rand()/(float)RAND_MAX;
      b[i] = (float)rand()/(float)RAND_MAX;
    }

  clock_t start = clock();
  struct timespec begin, end;
  clock_gettime(CLOCK_MONOTONIC_RAW, &begin);

  matrix_multiply_avx2(a, b, c);
  clock_gettime(CLOCK_MONOTONIC_RAW, &end);
    
  printf ("Total time = %f seconds\n",
          (end.tv_nsec - begin.tv_nsec) / 1000000000.0 +
          (end.tv_sec  - begin.tv_sec));
}
