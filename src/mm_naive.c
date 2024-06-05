#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>

typedef struct {
  float *a;
  float *b;
  float *c;
  int start_r;
  int end_r;
  int n;
} ThreadData;

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

void _mm_threads(void *arg) {
  ThreadData *data = (ThreadData*) arg;
  float *a = data->a;
  float *b = data->b;
  float *c = data->c;
  int start_r = data->start_r;
  int end_r = data->end_r;
  int n = data->n;

  for (int i=start_r; i<end_r; ++i)
    for (int k=0; k<n; ++k)
      for (int j=0; j<n; ++j)
        c[i * n + j] += a[i * n + k] * b[k * n + j];

  pthread_exit(NULL);
}

void mm_threads(float *a, float *b, float *c, int n, int n_threads) {
  pthread_t threads[n_threads];
  ThreadData args[n_threads];
  int rows_per_thread = n / n_threads;

  for (int i=0; i<n_threads; ++i) {
    args[i].a = a;
    args[i].b = b;
    args[i].c = c;
    args[i].start_r = i * rows_per_thread;
    args[i].end_r = args[i].start_r + rows_per_thread;
    args[i].n = n;

    if (pthread_create(&threads[i], NULL, _mm_threads, (void*)&args[i]) != 0) {
      perror("Failed to create thread");
      exit(1);
    }
  }

  for (int i=0; i<n_threads; ++i) {
    pthread_join(threads[i], NULL);
  }
}

