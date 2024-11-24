# Windows Env Testing Errors and Documentation

## Environment Details:

- **Operating System:** Windows 11 Home (Version 22H2)
- **Python Version:** 3.10.15

---

## 1: Installing fairseq-sl

### Steps to Reproduce:

1. Clone the repository:
   ```bash
   git clone https://github.com/facebookresearch/ssvp_slt
   ```
2. Create and activate a conda environment:
   ```bash
   conda create --name ssvp_slt python=3.10 cmake
   conda activate ssvp_slt
   ```
3. Move into fairseq folder and install egg:
   ```bash
   cd fairseq-sl
   pip install -e .
   ```

### Error Message 1:

```bash
error: subprocess-exited-with-error

× Getting requirements to build editable did not run successfully.
│ exit code: 1
╰─> See above for output.

note: This error originates from a subprocess, and is likely not a problem with pip.
```

### Suggested solution:

**Run Cmd as administrator**

### Error Message 2:

```bash
ERROR: Cannot install fairseq and fairseq==0.12.2 because these package versions have conflicting dependencies.
```

### Suggested solution:

```bash
python -m pip install pip<24.1
pip install omegaconf==2.0.5
```

---

## 2: Installing dlib with CUDA

### Steps to Reproduce:

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

### Error Message:

```bash
-- *** cuDNN V5.0 OR GREATER NOT FOUND.                                                       ***
-- *** Dlib requires cuDNN V5.0 OR GREATER.  Since cuDNN is not found DLIB WILL NOT USE CUDA. ***
-- *** If you have cuDNN then set CMAKE_PREFIX_PATH to include cuDNN's folder.                ***
-- Disabling CUDA support for dlib.  DLIB WILL NOT USE CUDA
```

### Workaround:

for now, no Suggested solution solved this error, but you caan run it on cpu or on a cloud.

---

## 3: Fetching extracted features

### Steps to Reproduce:

1. Run the evaluation script:
   ```bash
   python tests/translation_demo.py
   ```

### Error Message:

```bash
RuntimeError: Failed to fetch features after 10 retries.
```
