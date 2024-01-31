#include <cilk.h>
#include <stdio.h>

// Function to multiply two matrices
void matmul(int m1[4096][4096], int m2[4096][4096], int result[4096][4096]) {
  cilk_for (int i = 0; i < 4096; i++) {
    for (int j = 0; j < 4096; j++) {
      result[i][j] = 0;
      for (int k = 0; k < 4096; k++) {
        result[i][j] += m1[i][k] * m2[k][j];
      }
    }
  }
}

int main() {
  int m1[4096][4096], m2[4096][4096], result[4096][4096];
  
  // Initialize the matrices
  for (int i = 0; i < 4096; i++) {
    for (int j = 0; j < 4096; j++) {
      m1[i][j] = i * j;
      m2[i][j] = i + j;
    }
  }
  
  // Multiply the matrices using the cilk_for loop
  matmul(m1, m2, result);
  
  // Print the result matrix
  for (int i = 0; i < 4096; i++) {
    for (int j = 0; j < 4096; j++) {
      printf("%d ", result[i][j]);
    }
    printf("\n");
  }
  
  return 0;
}
