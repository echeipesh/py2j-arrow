import pyarrow as pa
import os


path = "/Users/eugene/tmp/tensor-out.np"
read_mmap = pa.memory_map(path, mode='r')
tensor = pa.read_tensor(read_mmap)
print(tensor)
print(tensor.to_numpy())
