import torch
import os

os.environ['CUDA_VISIBLE_DEVICES'] = "0"


if torch.cuda.is_available():
    print('GPU is available')
else:
    print('GPU is not available')


# show available GPU statistics
print("torch.cuda.device_count(): %d" % torch.cuda.device_count())
print("torch.cuda.memory_allocated: %fGB" % (torch.cuda.memory_allocated(0) / 1024 / 1024 / 1024))
print("torch.cuda.memory_reserved: %fGB" % (torch.cuda.memory_reserved(0) / 1024 / 1024 / 1024))
print("torch.cuda.max_memory_reserved: %fGB" % (torch.cuda.max_memory_reserved(0) / 1024 / 1024 / 1024))

import subprocess

p = subprocess.check_output('nvidia-smi')
print(p.decode("utf-8"))

