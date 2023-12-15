import torch
import numpy as np
import time

if torch.backends.mps.is_available():
    mps_device = torch.device("mps")

torch_rand1 = torch.rand(100, 100, 100, 100).to(mps_device)
torch_rand2 = torch.rand(100, 100, 100, 100).to(mps_device)
np_rand1 = torch.rand(100, 100, 100, 100)
np_rand2 = torch.rand(100, 100, 100, 100)

# GPU (Metal).
start_time = time.time()
# @ symbol is the matrix multiplication operator in pytorch.
rand = (torch_rand1 @ torch_rand2)
end_time = time.time()
elapsed_time = end_time - start_time
print(f"GPU elapsed time: {elapsed_time:.10f} seconds")

# CPU.
start_time = time.time()
rand = np.multiply(np_rand1, np_rand2)
end_time = time.time()
elapsed_time = end_time - start_time
print(f"CPU elapsed time: {elapsed_time:.10f} seconds")