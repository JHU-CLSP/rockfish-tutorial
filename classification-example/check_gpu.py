import torch

# show available GPU
print(torch.cuda.device_count())

if torch.cuda.is_available():
    print('GPU is available')
else:
    print('GPU is not available')
