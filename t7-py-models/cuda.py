import torch

if torch.cuda.is_available():
    print("CUDA está disponible y PyTorch puede usar la GPU.")
else:
    print("CUDA no está disponible o PyTorch no puede usar la GPU.")
