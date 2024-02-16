import torch

t = torch.randn(2, 2)
print(t.is_cuda)  # Returns False

t_gpu = torch.randn(2, 2).cuda()
print(t_gpu.is_cuda)  # Returns True

t_cpu = t_gpu.cpu()
print(t_cpu.is_cuda)  # Returns False
