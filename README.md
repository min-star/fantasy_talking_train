[中文阅读](./README_zh.md)

## Quickstart
### 🛠️Installation

Clone the repo:

```
git clone [https://github.com/Fantasy-AMAP/fantasy-talking.git](https://github.com/min-star/fantasy_talking_train)
cd fantasy-talking
```

Install dependencies:
```
# Ensure torch >= 2.0.0
pip install -r requirements.txt
# Optional to install flash_attn to accelerate attention computation
pip install flash_attn
```

### 🧱Model Download
| Models        |                       Download Link                                           |    Notes                      |
| --------------|-------------------------------------------------------------------------------|-------------------------------|
| Wan2.1-I2V-14B-720P  |      🤗 [Huggingface](https://huggingface.co/Wan-AI/Wan2.1-I2V-14B-720P)    🤖 [ModelScope](https://www.modelscope.cn/models/Wan-AI/Wan2.1-I2V-14B-720P)     | Base model
| Wav2Vec |      🤗 [Huggingface](https://huggingface.co/facebook/wav2vec2-base-960h)    🤖 [ModelScope](https://modelscope.cn/models/AI-ModelScope/wav2vec2-base-960h)      | Audio encoder
| FantasyTalking model      |      🤗 [Huggingface](https://huggingface.co/acvlab/FantasyTalking/)     🤖 [ModelScope](https://www.modelscope.cn/models/amap_cvlab/FantasyTalking/)         | Our audio condition weights

Download models using huggingface-cli:
``` sh
pip install "huggingface_hub[cli]"
huggingface-cli download Wan-AI/Wan2.1-I2V-14B-720P --local-dir ./models/Wan2.1-I2V-14B-720P
huggingface-cli download facebook/wav2vec2-base-960h --local-dir ./models/wav2vec2-base-960h
huggingface-cli download acvlab/FantasyTalking fantasytalking_model.ckpt --local-dir ./models
```

Download models using modelscope-cli:
``` sh
pip install modelscope
modelscope download Wan-AI/Wan2.1-I2V-14B-720P --local_dir ./models/Wan2.1-I2V-14B-720P
modelscope download AI-ModelScope/wav2vec2-base-960h --local_dir ./models/wav2vec2-base-960h
modelscope download amap_cvlab/FantasyTalking   fantasytalking_model.ckpt  --local_dir ./models
```

### 🔑 Inference
``` sh
python infer.py  --image_path ./assets/images/woman.png --audio_path ./assets/audios/woman.wav
```
You can control the character's behavior through the prompt. **The recommended range for prompt and audio cfg is [3-7]. You can increase the audio cfg to achieve more consistent lip-sync.**
``` sh
python infer.py  --image_path ./assets/images/woman.png --audio_path ./assets/audios/woman.wav --prompt "The person is speaking enthusiastically, with their hands continuously waving." --prompt_cfg_scale 5.0 --audio_cfg_scale 5.0
```

### Train
1. Prepare the Dataset
```
data2/train
├── 000.mp4
├── 000.wav
├── 005.mp4
├── 005.wav
└── metadata.csv
```
`metadata.csv` is the metadata list, for example:
```
file_name,text,audio
000.mp4,"A girl wearing a mask faces the audience and gently places her hands on her chest",000.wav
005.mp4,"A girl wearing a mask stands in the kitchen facing the audience and makes a heart shape with her hands",005.wav
```
2. Train
data to tensor.pt
```
CUDA_VISIBLE_DEVICES=0 python ./fantasytalking_train.py \
    --task data_process \
    --dataset_path "data1/" \
    --output_path "./models" \
    --text_encoder_path "./models/Wan2.1-I2V-14B-720P/models_t5_umt5-xxl-enc-bf16.pth" \
    --dit_path "models/Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00001-of-00007.safetensors,models/Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00002-of-00007.safetensors,models/Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00003-of-00007.safetensors,models/Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00004-of-00007.safetensors,models/Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00005-of-00007.safetensors,models/Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00006-of-00007.safetensors,models/Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00007-of-00007.safetensors" \
    --image_encoder_path "models/Wan2.1-I2V-14B-720P/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth" \
    --vae_path "./models/Wan2.1-I2V-14B-720P/Wan2.1_VAE.pth" \
    --tiled \
    --num_frames 81 \
    --height 480 \
    --width 832
```
single GPU
```
CUDA_VISIBLE_DEVICES=0 python ./fantasytalking_train.py \
    --task data_process \
    --dataset_path "data1/" \
    --output_path "./models" \
    --text_encoder_path "./models/Wan2.1-I2V-14B-720P/models_t5_umt5-xxl-enc-bf16.pth" \
    --dit_path "models/Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00001-of-00007.safetensors,models/Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00002-of-00007.safetensors,models/Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00003-of-00007.safetensors,models/Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00004-of-00007.safetensors,models/Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00005-of-00007.safetensors,models/Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00006-of-00007.safetensors,models/Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00007-of-00007.safetensors" \
    --image_encoder_path "models/Wan2.1-I2V-14B-720P/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth" \
    --vae_path "./models/Wan2.1-I2V-14B-720P/Wan2.1_VAE.pth" \
    --tiled \
    --num_frames 81 \
    --height 480 \
    --width 832
```
multi GPU
```
CUDA_VISIBLE_DEVICES="0,1,2,3" python ./fantasytalking_train.py     --task train     --dataset_path data1/     --output_path ./models     --dit_path "models/Wan2.1-I2V-14B-
720P/diffusion_pytorch_model-00001-of-00007.safetensors,models/Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00002-of-00007.safetensors,models/Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00003-of-00007.safetensors,models/Wan2.1-I2V-14B-720P/diffusion_pytorc
h_model-00004-of-00007.safetensors,models/Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00005-of-00007.safetensors,models/Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00006-of-00007.safetensors,models/Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00007-of-0000
7.safetensors"     --steps_per_epoch 500     --max_epochs 10     --learning_rate 1e-4     --accumulate_grad_batches 1     --use_gradient_checkpointing --use_gradient_checkpointing_offload --training_strategy "deepspeed_stage_2"
```
GPU 
A100 usage ~65GB
