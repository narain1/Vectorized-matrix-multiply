#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>

#define n 2048

float a[n][n], b[n][n], c[n][n];

int main() {
  time_t t;
  srand((unsigned) time(&t));
  for (int i=0; i<n;i++) 
    for (int j=0; j<n; j++) {
      a[i][j] = (float)rand()/(float)RAND_MAX;
      b[i][j] = (float)rand()/(float)RAND_MAX;
      c[i][j] = 0;
    }

  clock_t start = clock();
  struct timespec begin, end;
  clock_gettime(CLOCK_MONOTONIC_RAW, &begin);

  #pragma omp parallel for private(i,j,k) shared(a,b,c)
  for(int i=0; i<n; i++) 
    for(int j=0; j<n; j++)
      for(int k=0; k<n; k++)
        c[i][j] += a[i][k] * a[k][j];

  clock_gettime(CLOCK_MONOTONIC_RAW, &end);
    
  printf ("Total time = %f seconds\n",
          (end.tv_nsec - begin.tv_nsec) / 1000000000.0 +
          (end.tv_sec  - begin.tv_sec));
}
