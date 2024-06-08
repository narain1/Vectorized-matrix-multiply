#include <immintrin.h>

inline void sgemm_simd_block(
    const int M,
    const int N,
    const int K,
    const float *A,
    const float *B,
    float *C
    ) {
  const int nc = 4;
  const int kc = 240;
  const int mc = 120;
  const int nr = 4 * 8;
  const int mr = 3;

  for (int jc=0; jc<N; jc+=nc) {
    for (int pc=0; pc<K; pc+=kc) {
      for (int ic=0; ic<M; ic+=mc) {
        for (int jr=0; jr<nc; jr+=nr) {
          for (int ir=0; ir<mc; ir+=mr) {
            matmul_dot_inner_block(a,b,c,M,N,K,jc,nc,pc,kc,ic,mc,jr,nr,ir,mr);
          }
        }
      }
    }
  }
}
