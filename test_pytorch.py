import torch
import sys
import cv2

print(sys.version)
print(torch.__version__)
print(cv2.__version__)

print(torch.cuda.is_available())
print(torch.cuda.device_count())
print(torch.cuda.get_device_name(0))
print(torch.cuda.current_device())
print(torch.cuda.memory_allocated())
print(torch.cuda.memory_cached())
print(torch.cuda.memory_summary())
print(torch.cuda.max_memory_allocated())
print(torch.cuda.max_memory_cached())
print(torch.cuda.synchronize())
print(torch.cuda.set_device(0))
print(torch.cuda.get_device_capability(0))
print(torch.cuda.get_device_properties(0))
