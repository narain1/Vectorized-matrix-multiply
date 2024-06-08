#include <immintrin.h>

#define AddFloat8(PTR, VAL) _mm256_store_ps((PTR), _mm256_add_ps(_mm256_load_ps(PTR), (VAL)))

void mm_kernel(
    const float * __restrict__ a,
    const float * __restrict__ b,
    float *c,
    const int m, const int n, const int k,
    const int jc, const int nc,
    const int pc, const int kc,
    const int ic, const int mc,
    const int jr, const int nr,
    const int ir, const int mr)
{
  __m256 csum[6][2] = {{ _mm256_set1_ps(0) }};
  for (int k=0; k < kc; k++) {
    for (unsigned ai=0; ai < 6; ai++) {
      __m256 aa = _mm256_set1_ps(a[(ic + ir * ai) * k * pc + k]);
      for (unsigned bi=0; bi<2; bi++) {
        __m256 bb = _mm256_load_ps(&b[(pc + k) * n + jc + jr + bi * 8]);
        csum[ai][bi] = _mm256_fmadd_ps(aa, bb, csum[ai][bi]);
      }
    }
  }

  for (unsigned ai=0; ai<6; ai++) {
    for (unsigned bi=0; bi<2; bi++) {
      AddFloat8(&c[(ic + ir + ai) * n + jc + jr + bi * 8], csum[ai][bi]);
    }
  }
}

void mm(const float *a, const float *b, float *c, const int m, const int n, const int k)
{
  const int nc = n;
  const int kc = 240;
  const int mc = 120;
  const int nr = 2 * 8;
  const int mr = 6;

  for (int jc=0; jc<nc; jc+=nc) {
    for (int pc=0; pc<k; pc+=kc) {
      for (int ic=0; ic<m; ic+=mc) {
        for (int jr=0; jr<nc; jr+=nr) {
          for (int ir=0; ir<mc; ir+=mr) {
            mm_kernel(a, b, c, m, n, k, jc, nc, pc, kc, ic, mc, jr, nr, ir, mr);
          }
        }
      }
    }
  }
}
