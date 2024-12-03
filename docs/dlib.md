# Installing dlib with CUDA on windows

## Environment Details:

- **Operating System:** Windows 11 Home (Version 22H2)
- **Python Version:** 3.10.15

## Steps

1. activate a conda environment:
   ```bash
   conda activate ssvp_slt
   ```
2. Install CUDA and cuDNN:
   ```bash
   conda install cuda cudnn -c nvidia
   ```
3. Clone the repository:

   ```bash
   git clone https://github.com/davisking/dlib.git
   ```

4. make build dir:

   ```bash
   cd dlib
   mkdir build
   cd build
   ```

5. Install dlib from the source:
   ```bash
   cmake .. -DDLIB_USE_CUDA=1 -DUSE_AVX_INSTRUCTIONS=1 -DCUDAToolkit_ROOT=/path/to/your/conda/envs/dlib/bin/ -G "Visual Studio 17 2022" -A x64
   cmake --build .
   cd ..
   python setup.py install --set DLIB_USE_CUDA=1
   ```

---
