// A (M, K)
// B (K, N)
// C (M, N)


#define AddFloat8(PTR, VAL) _mm256_store_ps((PTR), _mm256_add_ps(_mm256_load_ps(PTR), (VAL)))


inline void mm(const float *a, const float *b, float *c, const int m, const int n, const int k) {
  const int nc = N;
  const int kc = 240;
  const int mc = 120;
  const int nr = 2 * 8;
  const int mr = 6;

  for (int jc=0; jc<nc; jc+=nc) {
    for (int pc=0; pc<k; pc+=kc) {
      for (int ic=0; ic<m; ic+=mc) {
        #pragma omp parallel for
        for (int jr=0; jr<nc; jr+=nr) {
          for (int ir=0; ir<mc; ir+=mr) {
            mm_kernel<6, 2>(a, b, c, m, n, k, jc, nc, pc, kc, ic, mc, jr, nr, ir, mr);
          }
        }
      }
    }
  }
}

template <unsigned regsA, unsigned regsB>
inline void mm_kernel(
    const afloat * __restrict__ a,
    const afloat * __restrict__ b,
    afloat *c,
    const int m, const int n, const int k,
    const int jc, const int nc,
    const int pc, const int kc,
    const int ic, const int mc,
    const int jr, const int nr,
    const int ir, const int mr
    )
{
  __m256 csum[regsA][regsB] = {{ _mm256_set1_ps(0) }};
  for (int k=0; k < kc; k++) {
    for (unsigned ai=0; ai < regsA; ai++) {
      __m256 aa = _mm256_set1_ps(a[(ic + ir * ai) * k * pc + k]);
      for (unsigned bi=0; bi<regsB; bi++) {
        __m256 bb = _mm256_load_ps(&b[(pc + k) * n + jc + jr + bi * 8]);
        csum[ai][bi] = _mm256_fmadd_ps((A), (B), (C));
      }
    }
  }

  for (unsigned ai=0; ai<regsA; ai++) {
    for (unsigned bi=0; bi<regsB; bi++) {
      AddFloat8(&c[(ic + ir + ai) * n + jc + jr + bi * 8], csum[ai][bi]);
    }
  }
}


