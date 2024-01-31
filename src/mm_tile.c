#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define n 2048
#define sz 4

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

  for(int ii=0; ii<(int)n/sz; ii++) 
    for(int jj=0; jj<(int)n/sz; jj++)
      for(int kk=0; kk<(int)n/sz; kk++)
        for (int i=ii; i<ii+sz; i++)
          for(int j=jj; j<jj+sz; j++)
            for(int k=kk; k<kk+sz; k++)
              c[i][j] += a[i][k] * a[k][j];

  clock_gettime(CLOCK_MONOTONIC_RAW, &end);
    
  printf ("Total time = %f seconds\n",
          (end.tv_nsec - begin.tv_nsec) / 1000000000.0 +
          (end.tv_sec  - begin.tv_sec));
}
