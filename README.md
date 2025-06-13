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

## 🧩 Community Works
We ❤️ contributions from the open-source community! If your work has improved FantasyTalking, please inform us.
Or you can directly e-mail [frank.jf@alibaba-inc.com](mailto://frank.jf@alibaba-inc.com). We are happy to reference your project for everyone's convenience.

## 🔗Citation
If you find this repository useful, please consider giving a star ⭐ and citation
```
@article{wang2025fantasytalking,
   title={FantasyTalking: Realistic Talking Portrait Generation via Coherent Motion Synthesis},
   author={Wang, Mengchao and Wang, Qiang and Jiang, Fan and Fan, Yaqi and Zhang, Yunpeng and Qi, Yonggang and Zhao, Kun and Xu, Mu},
   journal={arXiv preprint arXiv:2504.04842},
   year={2025}
 }
```

## Acknowledgments
Thanks to [Wan2.1](https://github.com/Wan-Video/Wan2.1), [HunyuanVideo](https://github.com/Tencent/HunyuanVideo), and [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio) for open-sourcing their models and code, which provided valuable references and support for this project. Their contributions to the open-source community are truly appreciated.
