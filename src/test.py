import ctypes
import numpy as np
import time

# Load the shared library
lib = ctypes.CDLL('./lib.so')

# Define the function types
lib.mm_naive.argtypes = [np.ctypeslib.ndpointer(dtype=np.float32, ndim=2, flags='C_CONTIGUOUS'),
                         np.ctypeslib.ndpointer(dtype=np.float32, ndim=2, flags='C_CONTIGUOUS'),
                         np.ctypeslib.ndpointer(dtype=np.float32, ndim=2, flags='C_CONTIGUOUS'),
                         ctypes.c_int]
lib.mm_naive.restype = None

lib.mm_transpose.argtypes = [np.ctypeslib.ndpointer(dtype=np.float32, ndim=2, flags='C_CONTIGUOUS'),
                         np.ctypeslib.ndpointer(dtype=np.float32, ndim=2, flags='C_CONTIGUOUS'),
                         np.ctypeslib.ndpointer(dtype=np.float32, ndim=2, flags='C_CONTIGUOUS'),
                         ctypes.c_int]
lib.mm_transpose.restype = None

lib.mm_threads.argtypes = [np.ctypeslib.ndpointer(dtype=np.float32, ndim=2, flags='C_CONTIGUOUS'),
                         np.ctypeslib.ndpointer(dtype=np.float32, ndim=2, flags='C_CONTIGUOUS'),
                         np.ctypeslib.ndpointer(dtype=np.float32, ndim=2, flags='C_CONTIGUOUS'),
                         ctypes.c_int]
lib.mm_threads.restype = None

lib.mm_tiled_omp.argtypes = [np.ctypeslib.ndpointer(dtype=np.float32, ndim=2, flags='C_CONTIGUOUS'),
                         np.ctypeslib.ndpointer(dtype=np.float32, ndim=2, flags='C_CONTIGUOUS'),
                         np.ctypeslib.ndpointer(dtype=np.float32, ndim=2, flags='C_CONTIGUOUS'),
                         ctypes.c_int]
lib.mm_tiled_omp.restype = None

# Initialize matrices
n = 1024
a = np.random.randn(n, n).astype(np.float32)
b = np.random.randn(n, n).astype(np.float32)
c = np.zeros((n, n), dtype=np.float32)

# Call the function
start = time.perf_counter()
lib.mm_naive(a, b, c, n)
end = time.perf_counter()

print("time taken mm_naive : ", end - start)

start = time.perf_counter()
lib.mm_transpose(a, b, c, n)
end = time.perf_counter()

print("time taken mm transpose: ", end - start)

start = time.perf_counter()
lib.mm_threads(a, b, c, n)
end = time.perf_counter()

print("time taken mm threads: ", end - start)

start = time.perf_counter()
lib.mm_tiled_omp(a, b, c, n)
end = time.perf_counter()

print("time taken mm threads: ", end - start)
