#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>

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

void _mm_threads(float *a, float *b, float *c, int start_r, int end_r, int n) {
  int i,j,k;
  for (i=start_r; i<end_r; ++i)
    for (k=0; k<n; ++k)
      for (j=0; j<n; ++j)
        c[i * n + j] += a[i * n + k] * b[k * n + j];
  pthread_exit(NULL);
}

void mm_threads(float *a, float *b, float *c, int n, int n_threads) {
  pthread_t threads[n_threads];
  int rows_per_thread = n / n_threads;
  int rem_rows = n % n_threads;

  int i;
  for (i=0; i<n_threads; ++i) {
    start_r = i * rows_per_thread;
    end_r = start_r + rows_per_thread;
    int rc = pthread_create(&threads[i], NULL, multiply, (void*)&args[i]);
    

