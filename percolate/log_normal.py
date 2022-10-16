import torch

LOG_NORMAL_ZERO_THRESHOLD = 1e-5
pi_val = torch.acos(torch.zeros(1)).item() * 2