import torch
import time
a = torch.randn(2, 3).cuda()

print(a.shape)

time.sleep(100)