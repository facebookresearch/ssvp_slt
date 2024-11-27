

### Installation

1. Create and activate conda environment
```bash
conda create --name sign-language-inference python=3.10
conda activate sign-language-inference
```

2. Install dependencies
```bash
conda install -c -y conda-forge libsndfile==1.0.31 # fairseq2 dependency
pip install -r requirements.txt
```

3. Install dlib with CUDA support (alternatively, `pip install dlib`; comes without CUDA support)

Confirm whether your dlib installation supports CUDA with: `python -c "import dlib; print(dlib.DLIB_USE_CUDA)"`.

```bash
# This configuration worked on FAIR cluster; make sure to check build logs

module load cmake/3.15.3/gcc.7.3.0 gcc/12.2.0 cuda/12.1 cudnn/v8.8.1.3-cuda.12.0

git clone https://github.com/davisking/dlib.git
cd dlib
python setup.py install \
  --set CUDA_TOOLKIT_ROOT_DIR=/public/apps/cuda/12.1 \
  --set CMAKE_PREFIX_PATH=/public/apps/cudnn/v8.8.1.3-cuda.12.0 \
  --set USE_AVX_INSTRUCTIONS=yes \
  --set DLIB_USE_CUDA=yes

```

4. Install this repo (from repo root)
```bash
pip install -e .
```


### Model checkpoints

Copy model checkpoints to [checkpoints](checkpoints) folder.
```bash
cp -r /checkpoint/philliprust/slt_inference/checkpoints/* checkpoints/
```

### Running Inference

```bash
python run.py video_path=/path/to/video.mp4
```

Pass `verbose=True` to print config and time elapsed for various steps in the pipeline.

Pass `preprocessing.hog_detector=false` if running dlib CNN detector with CUDA support.

Pass `preprocessing.detection_downsample=false` if the video input resolution is already small, e.g. 224x224.

Here is a more advanced example for running a SONAR model:

```bash
python run.py \
  video_path=/path/to/my/video.mp4 \
  use_sonar=true \
  preprocessing.detection_downsample=false \
  feature_extraction.pretrained_model_path=/path/to/trained/signhiera.pth \
  translation.base_model_name=google/t5-v1_1-large \
  translation.pretrained_model_path=/path/to/trained/sonar/encoder/best_model.pth \
  feature_extraction.fp16=true \
  'translation.tgt_langs=[eng_Latn, fra_Latn, deu_Latn, zho_Hans]'
```


### Running the Gradio app

To run the Gradio app, you can use the exact same command as for `run.py` but leave out the `video_path`. You can then open the browser and upload a video. As soon as the video finishes uploading, it should start playing automatically which will trigger the inference pipeline. If the video doesn't start automatically, just press the play button manually.

```bash
python app.py \
  use_sonar=true \
  preprocessing.detection_downsample=false \
  feature_extraction.pretrained_model_path=/path/to/trained/signhiera.pth \
  translation.base_model_name=google/t5-v1_1-large \
  translation.pretrained_model_path=/path/to/trained/sonar/encoder/best_model.pth \
  feature_extraction.fp16=true \
  'translation.tgt_langs=[eng_Latn, fra_Latn, deu_Latn, zho_Hans]'
```

The Gradio app needs to be launched from a GPU machine. If running on a devfair, you can tunnel to your local machine via `ssh -L 8000:localhost:7860 devfair`. After the devfair login, you can go to `localhost:8000` in your browser to use the app.

### Slicing a video before inferencing

Videos should ideally be a sentence long. To slice a video, e.g. from second 10 to 14, you can use ffmpeg:

```bash
ffmpeg -ss 10.00 -to 14.00 -i ./video.MOV -c:v libx264 -crf 20 video_slice.mp4 -loglevel info
```