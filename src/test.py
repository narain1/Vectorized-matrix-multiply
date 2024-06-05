import ctypes

lib = ctypes.CDLL("./lib.so")

print("loaded lib")
