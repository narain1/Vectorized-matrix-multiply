import numpy as np
import ctypes

# Load the shared library
lib = ctypes.CDLL('./lib.so')

# Define the matrix dimensions
M = N = K = 960

# Create matrices
a = np.random.rand(M, K).astype(np.float32)
b = np.random.rand(K, N).astype(np.float32)
c = np.zeros((M, N), dtype=np.float32)

# Define the types for the arguments for safety
lib.matmul_dot_inner.argtypes = [
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=2, flags='C_CONTIGUOUS'),
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=2, flags='C_CONTIGUOUS'),
    np.ctypeslib.ndpointer(dtype=np.float32, ndim=2, flags='C_CONTIGUOUS')
]

# Call the matrix multiplication function
lib.matmul_dot_inner(M, N, K, a, b, c)

# Now `c` contains the result of the matrix multiplication
print(c)

