# Copyright Alibaba Inc. All Rights Reserved.

import imageio
import librosa
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm


def resize_image_by_longest_edge(image_path, target_size):
    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    scale = target_size / max(width, height)
    new_size = (int(width * scale), int(height * scale))
    return image.resize(new_size, Image.LANCZOS)


def save_video(frames, save_path, fps, quality=9, ffmpeg_params=None):
    writer = imageio.get_writer(
        save_path, fps=fps, quality=quality, ffmpeg_params=ffmpeg_params
    )
    for frame in tqdm(frames, desc="Saving video"):
        frame = np.array(frame)
        writer.append_data(frame)
    writer.close()


def get_audio_features(wav2vec, audio_processor, audio_path, fps, num_frames):  # fps: 23 num_frames: 81
    sr = 16000
    audio_input, sample_rate = librosa.load(audio_path, sr=sr)  # 采样率为 16kHz  (160125,) 16000

    start_time = 0
    # end_time = (0 + (num_frames - 1) * 1) / fps
    end_time = num_frames / fps  # 3.5217391304347827

    start_sample = int(start_time * sr)  # 0
    end_sample = int(end_time * sr)  # 56347

    try:
        audio_segment = audio_input[start_sample:end_sample]  # 截取音频片段 (56347,)
    except:
        audio_segment = audio_input

    input_values = audio_processor(
        audio_segment, sampling_rate=sample_rate, return_tensors="pt"  
    ).input_values.to("cuda")  # 输入数组版的音频片段 (56347，) 和采样率 16000 结果 [1, 56347]

    with torch.no_grad():
        fea = wav2vec(input_values).last_hidden_state  # 取出最后一层的隐藏状态 (1, 175, 768)

    return fea
