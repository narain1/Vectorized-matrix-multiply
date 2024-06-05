#include <stdio.h>
#include <stdlib.h>
#include <omp.h>


void mm_naive(float *a, float *b, float *c, int n)
{
  for(int i=0; i < n; i++)
    for (int j=0; j < n; j++)
      for (int k=0; k < n; k++)
        c[i * n + j] += a[i * n + k] * b[k * n + j];
}

void mm_transpose(float *a, float *b, float *c, int n)
{
  for(int i=0; i < n; i++)
    for (int k=0; k< n; k++)
      for (int j=0; j< n; j++)
        c[i * n + j] += a[i * n + k] * b[k * n + j];
}

void mm_threads(float *a, float *b, float *c, int n) {
  int i,j,k;
  #pragma omp parallel for private(i,j,k) shared(a,b,c)
  for(i=0; i < n; i++)
    for (k=0; k< n; k++)
      for (j=0; j< n; j++)
        c[i * n + j] += a[i * n + k] * b[k * n + j];
}

