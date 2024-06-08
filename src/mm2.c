#include <immintrin.h>


void matmul_dot_inner(
    const float *a,
    const float *b,
    float *c,
    const int M, const int N, const int K,
    const int m, const int n)
{
  __m256 csum[3][4] = {{ _mm256_set1_ps(0) }};
  for (int k=0; k<K; k++) {
    for (unsigned ai=0; ai<3; ai++) {
      __m256 aa = _mm256_set1_ps(a[(m+ai) * K + k]);
      for (unsigned bi=0; bi<4; bi++) {
        __m256 bb = _mm256_load_ps(&b[k*N + n + bi * 8]);
        csum[ai][bi] = _mm256_fmadd_ps(aa, bb, csum[ai][bi]);
      }
    }
  }

  for (unsigned ai=0; ai<3; ai++) {
    for (unsigned bi=0; bi<4; bi++) {
      *((__m256 *)(&c[(m+ai) * N+n+bi*8])) = csum[ai][bi];
    }
  }
}

void mm(const int M, const int N, const int K, const float *a, const float *b, float *c)
{
  for (int m=0; m<M; m+=3) {
    for (int n=0; n<N; n+=4) {
      matmul_dot_inner(a, b, c, M, N, K, m, n);
    }
  }
}
