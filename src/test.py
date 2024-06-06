import ctypes
import numpy as np
import time
import pandas as pd

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

lib.mm_vector.argtypes = [np.ctypeslib.ndpointer(dtype=np.float32, ndim=2, flags='C_CONTIGUOUS'),
                         np.ctypeslib.ndpointer(dtype=np.float32, ndim=2, flags='C_CONTIGUOUS'),
                         np.ctypeslib.ndpointer(dtype=np.float32, ndim=2, flags='C_CONTIGUOUS'),
                         ctypes.c_int]
lib.mm_vector.restype = None

# Initialize matrices
n = 1024

ops = ['mm_naive', 'mm_transpose', 'mm_threads', 'mm_tiled_omp', 'mm_vector']
time_acc = []
for op in ops:
    a = np.random.randn(n, n).astype(np.float32)
    b = np.random.randn(n, n).astype(np.float32)
    c = np.zeros((n, n), dtype=np.float32)

    start = time.perf_counter()
    getattr(lib, op)(a, b, c, n)
    end = time.perf_counter()
    time_acc.append(end - start)

table = pd.DataFrame({'method': ops, 'timings': time_acc})
table.to_markdown('../benchmark.md')
print(table)
