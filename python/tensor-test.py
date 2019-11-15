from py4j.java_gateway import JavaGateway
import numpy as np
import pyarrow as pa
import os


gateway = JavaGateway()

data = np.random.randn(10, 4)
tensor = pa.Tensor.from_numpy(data)
tmpdir = "/Users/eugene/tmp"
path = os.path.join(str(tmpdir), 'pyarrow-tensor-ipc-roundtrip')
mmap = pa.create_memory_map(path, 1024)
pa.write_tensor(tensor, mmap)
mmap.close()

# pa.write_tensor(tensor, mmap)
#
# arr = pa.ExtensionArray.from_storage(nd4j_type, pa.array([4, 40], pa.uint32()))
# batch = pa.RecordBatch.from_arrays([arr], ["eyes"])
# buf = ipc_write_batch(batch)
# gateway.entry_point.send_batch(buf.to_pybytes())
#
# eye_recv = gateway.entry_point.identity(4)
# print("received buffer of size", len(eye_recv))
#
# bos = pa.BufferOutputStream()
# tens = pa.Tensor.from_numpy(np.eye(4))
# pa.ipc.write_tensor(tens, bos)
# bos.close()
# byts = bos.getvalue().to_pybytes()
# gateway.entry_point.send(byts)
