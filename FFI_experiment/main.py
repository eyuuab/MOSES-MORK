# import ctypes
# import os

# lib = ctypes.CDLL(os.path.abspath("MyLib.so"), mode=ctypes.RTLD_GLOBAL)
# lib.MyLib_init()

# # 3. Define function signatures
# # Tell Python strictly what the C function expects
# lib.my_add.argtypes = [ctypes.c_int, ctypes.c_int]
# lib.my_add.restype = ctypes.c_int

# # 4. Call the function
# result = lib.my_add(1000, "4666")
# print(f"Haskell says: {result}")

# # 5. Clean up
# lib.MyLib_end()

import ctypes
import os

lib = ctypes.CDLL(os.path.abspath("MyLib.so"), mode=ctypes.RTLD_GLOBAL)
lib.MyLib_init()

# Define the signatures for the Quantale operations
lib.q_join_logical.argtypes = [ctypes.c_bool, ctypes.c_bool]
lib.q_join_logical.restype = ctypes.c_bool

lib.q_times_logical.argtypes = [ctypes.c_bool, ctypes.c_bool]
lib.q_times_logical.restype = ctypes.c_bool

# Test it
val1 = "1"
val2 = False

print(f"Join (OR): {lib.q_join_logical(val1, val2)}")   # Expected: True
print(f"Times (AND): {lib.q_times_logical(val1, val2)}") # Expected: False

lib.MyLib_end()