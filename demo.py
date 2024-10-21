import torch
import dlib
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.device_count())
print(torch.version.cuda)
print(dlib.DLIB_USE_CUDA)
print(dlib.cuda.get_num_devices())
