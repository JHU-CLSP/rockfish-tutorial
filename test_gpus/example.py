import torch
import os
import subprocess


def print_gpu_memory():
    print("torch.cuda.device_count(): %d" % torch.cuda.device_count())
    print("torch.cuda.memory_allocated: %fGB" % (torch.cuda.memory_allocated(0) / 1024 / 1024 / 1024))
    print("torch.cuda.memory_reserved: %fGB" % (torch.cuda.memory_reserved(0) / 1024 / 1024 / 1024))
    print("torch.cuda.max_memory_reserved: %fGB" % (torch.cuda.max_memory_reserved(0) / 1024 / 1024 / 1024))

    p = subprocess.check_output('nvidia-smi')
    print(p.decode("utf-8"))


os.environ['CUDA_VISIBLE_DEVICES'] = "0"


if torch.cuda.is_available():
    print('GPU is available')
else:
    print('GPU is not available')


# show available GPU statistics
print_gpu_memory()

# let's test how much memory we can take
# creat increasingly large tensors in gpu memory
all = []
while True:
    all.append(torch.rand(100000, 100000).cuda())
    print_gpu_memory()
    print(" - - - - - ")

# you should see that the memory allocated and reserved keeps increasing
# until you run out of memory at around 40GB
