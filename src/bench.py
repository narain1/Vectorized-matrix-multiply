import numpy as np
import ctypes

# Load the shared library
lib = ctypes.CDLL('./mm.so')

# Define argument types
lib.mm.argtypes = [
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=2, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=2, flags="C_CONTIGUOUS"),
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=2, flags="C_CONTIGUOUS"),
    ctypes.c_int, ctypes.c_int, ctypes.c_int
]

# Define matrices
x = 16 * 6 * 30
m, n, k = x, x, x
A = np.random.rand(m, k).astype(np.float32)
B = np.random.rand(k, n).astype(np.float32)
C = np.zeros((m, n), dtype=np.float32)

# Call the matrix multiplication function
print("start")
lib.mm(A, B, C, m, n, k)

# C now contains the result of A * B
print("Matrix C:", C)
