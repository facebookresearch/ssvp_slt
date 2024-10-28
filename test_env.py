import torch
import dlib
import os

# Check if CUDA is available and if it's using a GPU
if torch.cuda.is_available():
  print("CUDA is available!")
  print(torch.__version__)
  print(torch.version.cuda)
  print(f"Device count: {torch.cuda.device_count()}")
  print(f"Current device: {torch.cuda.current_device()}")
  print(f"Device name: {torch.cuda.get_device_name(0)}")

else:
  print("CUDA is not available.")
  # Consider using a runtime with GPU acceleration in Colab.
  print("Consider using a runtime with GPU acceleration in Colab.")


# If dlib CUDA is available, print details
if dlib.DLIB_USE_CUDA:
  print("dlib CUDA is available!")
  print(f"Number of dlib CUDA devices: {dlib.cuda.get_num_devices()}")
else:
  print("dlib CUDA is not available.")
  # Consider alternative libraries or configurations if needed.