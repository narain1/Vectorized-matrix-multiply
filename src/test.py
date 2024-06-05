import ctypes
import numpy as np

# Load the shared library
lib = ctypes.CDLL('./lib.so')

# Define the function types
lib.mm_naive.argtypes = [np.ctypeslib.ndpointer(dtype=np.float32, ndim=2, flags='C_CONTIGUOUS'),
                         np.ctypeslib.ndpointer(dtype=np.float32, ndim=2, flags='C_CONTIGUOUS'),
                         np.ctypeslib.ndpointer(dtype=np.float32, ndim=2, flags='C_CONTIGUOUS'),
                         ctypes.c_int]
lib.mm_naive.restype = None

# Initialize matrices
n = 1024
a = np.random.randn(n, n).astype(np.float32)
b = np.random.randn(n, n).astype(np.float32)
c = np.zeros((n, n), dtype=np.float32)

# Call the function
lib.mm_naive(a, b, c, n)

print("Result of matrix multiplication:")
print(c)
