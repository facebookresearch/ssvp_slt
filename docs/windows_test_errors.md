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
