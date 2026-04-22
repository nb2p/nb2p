import ctypes
import gc

del gc.garbage[:]
gc.collect()
libc = ctypes.CDLL("libc.so.6")
libc.malloc_trim(0)
print("GC done")
