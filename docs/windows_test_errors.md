# Windows Env Testing Errors and Documentation

## Environment Details:

- **Operating System:** Windows 11 Home (Version 22H2)
- **Python Version:** 3.10.15

---

## Error 1: Installing fairseq-sl

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

### Error Message:

```bash
ERROR: Cannot install fairseq and fairseq==0.12.2 because these package versions have conflicting dependencies.
```

### Traceback:

```bash
The conflict is caused by:
    fairseq 0.12.2 depends on omegaconf<2.1
    hydra-core 1.0.7 depends on omegaconf<2.1 and >=2.0.5
```

---

## Error 2: Installing dlib with CUDA

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

---

## Error 3: Fetching extracted features

### Steps to Reproduce:

1. Run the evaluation script:
   ```bash
   python tests/translation_demo.py
   ```

### Error Message:

```bash
RuntimeError: Failed to fetch features after 10 retries.
```

### Traceback:

```bash
Traceback (most recent call last):
  File "D:\Pro\MLH\SLT\tests\translation_demo.py", line 39, in <module>
    run_translation(translation_dict_config)
  File "D:\Pro\MLH\SLT\translation\run_translation_module.py", line 158, in run_translation
    translate(cfg)
  File "D:\Pro\MLH\SLT\translation\main_translation.py", line 37, in eval
    stats, predictions, references = evaluate(
  File "D:\Pro\MLH\conda\envs\ssvp_slt\lib\site-packages\torch\utils\_contextlib.py", line 115, in decorate_context
    return func(*args, **kwargs)
  File "D:\Pro\MLH\SLT\translation\engine_translation.py", line 208, in evaluate
    for data_iter_step, batch in enumerate(metric_logger.log_every(dataloader, 10, header)):
  File "d:\pro\mlh\slt\src\ssvp_slt\util\misc.py", line 207, in log_every
    for obj in iterable:
  File "D:\Pro\MLH\conda\envs\ssvp_slt\lib\site-packages\torch\utils\data\dataloader.py", line 631, in __next__
    data = self._next_data()
  File "D:\Pro\MLH\conda\envs\ssvp_slt\lib\site-packages\torch\utils\data\dataloader.py", line 675, in _next_data
    data = self._dataset_fetcher.fetch(index)  # may raise StopIteration
  File "D:\Pro\MLH\conda\envs\ssvp_slt\lib\site-packages\torch\utils\data\_utils\fetch.py", line 51, in fetch
    data = [self.dataset[idx] for idx in possibly_batched_index]
  File "D:\Pro\MLH\conda\envs\ssvp_slt\lib\site-packages\torch\utils\data\_utils\fetch.py", line 51, in <listcomp>
    data = [self.dataset[idx] for idx in possibly_batched_index]
  File "D:\Pro\MLH\conda\envs\ssvp_slt\lib\site-packages\torch\utils\data\dataset.py", line 335, in __getitem__
    data = self._dataset_fetcher.fetch(index)  # may raise StopIteration
  File "D:\Pro\MLH\conda\envs\ssvp_slt\lib\site-packages\torch\utils\data\_utils\fetch.py", line 51, in fetch
    data = [self.dataset[idx] for idx in possibly_batched_index]
  File "D:\Pro\MLH\conda\envs\ssvp_slt\lib\site-packages\torch\utils\data\_utils\fetch.py", line 51, in <listcomp>
    data = [self.dataset[idx] for idx in possibly_batched_index]
    data = self._dataset_fetcher.fetch(index)  # may raise StopIteration
  File "D:\Pro\MLH\conda\envs\ssvp_slt\lib\site-packages\torch\utils\data\_utils\fetch.py", line 51, in fetch
    data = [self.dataset[idx] for idx in possibly_batched_index]
  File "D:\Pro\MLH\conda\envs\ssvp_slt\lib\site-packages\torch\utils\data\_utils\fetch.py", line 51, in <listcomp>
    data = self._dataset_fetcher.fetch(index)  # may raise StopIteration
  File "D:\Pro\MLH\conda\envs\ssvp_slt\lib\site-packages\torch\utils\data\_utils\fetch.py", line 51, in fetch
    data = [self.dataset[idx] for idx in possibly_batched_index]
    data = self._dataset_fetcher.fetch(index)  # may raise StopIteration
  File "D:\Pro\MLH\conda\envs\ssvp_slt\lib\site-packages\torch\utils\data\_utils\fetch.py", line 51, in fetch
    data = [self.dataset[idx] for idx in possibly_batched_index]
  File "D:\Pro\MLH\conda\envs\ssvp_slt\lib\site-packages\torch\utils\data\_utils\fetch.py", line 51, in <listcomp>
    data = self._dataset_fetcher.fetch(index)  # may raise StopIteration
  File "D:\Pro\MLH\conda\envs\ssvp_slt\lib\site-packages\torch\utils\data\_utils\fetch.py", line 51, in fetch
    data = [self.dataset[idx] for idx in possibly_batched_index]
  File "D:\Pro\MLH\conda\envs\ssvp_slt\lib\site-packages\torch\utils\data\_utils\fetch.py", line 51, in <listcomp>
    data = self._dataset_fetcher.fetch(index)  # may raise StopIteration
  File "D:\Pro\MLH\conda\envs\ssvp_slt\lib\site-packages\torch\utils\data\_utils\fetch.py", line 51, in fetch
    data = self._dataset_fetcher.fetch(index)  # may raise StopIteration
    data = self._dataset_fetcher.fetch(index)  # may raise StopIteration
  File "D:\Pro\MLH\conda\envs\ssvp_slt\lib\site-packages\torch\utils\data\_utils\fetch.py", line 51, in fetch
    data = [self.dataset[idx] for idx in possibly_batched_index]
  File "D:\Pro\MLH\conda\envs\ssvp_slt\lib\site-packages\torch\utils\data\_utils\fetch.py", line 51, in <listcomp>
    data = [self.dataset[idx] for idx in possibly_batched_index]
  File "D:\Pro\MLH\conda\envs\ssvp_slt\lib\site-packages\torch\utils\data\dataset.py", line 335, in __getitem__
    return self.datasets[dataset_idx][sample_idx]
  File "d:\pro\mlh\slt\src\ssvp_slt\data\sign_features_dataset.py", line 158, in __getitem__
    raise RuntimeError(f"Failed to fetch features after {self.num_retries} retries.")
```
