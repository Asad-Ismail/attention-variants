import torch
import random
import numpy as np


def set_seed(seed=42, deterministic_cudnn=True):
    random.seed(seed)
    # NumPy
    np.random.seed(seed) 
    # PyTorch (both CPU and CUDA)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU setups
    # Optional: Make CuDNN deterministic (might slow down training)
    if deterministic_cudnn:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
