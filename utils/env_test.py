import torch
import torch.nn as nn
from torch.nn import functional as F

torch_version = torch.__version__
cuda_available = torch.cuda.is_available()

print(f'{torch_version = }')
print(f'{cuda_available = }')
