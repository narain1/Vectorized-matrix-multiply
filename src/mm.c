#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define TILE_SIZE 8  // Define the size of the tile

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

void mm_threads(float *a, float *b, float *c, int n)
{
  int i,j,k;
  // #pragma omp parallel for private(i,j,k) shared(a,b,c)
  #pragma omp parallel for collapse(2) if (n * n * n > 300000)
  for(i=0; i < n; i++)
    for (k=0; k< n; k++)
      for (j=0; j< n; j++)
        c[i * n + j] += a[i * n + k] * b[k * n + j];
}

void mm_tiled_omp(float *a, float *b, float *c, int n)
{
    int i, j, k, i1, j1, k1;

    #pragma omp parallel for private(i, j, k, i1, j1, k1) shared(a, b, c, n) schedule(dynamic)
    for (i = 0; i < n; i += TILE_SIZE) {
        for (j = 0; j < n; j += TILE_SIZE) {
            for (k = 0; k < n; k += TILE_SIZE) {
                // Process a block/tile
                for (i1 = i; i1 < i + TILE_SIZE && i1 < n; i1++) {
                    for (j1 = j; j1 < j + TILE_SIZE && j1 < n; j1++) {
                        float sum = 0.0f;
                        for (k1 = k; k1 < k + TILE_SIZE && k1 < n; k1++) {
                            sum += a[i1 * n + k1] * b[k1 * n + j1];
                        }
                        #pragma omp atomic
                        c[i1 * n + j1] += sum;
                    }
                }
            }
        }
    }
}
