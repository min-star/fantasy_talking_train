# Copyright Alibaba Inc. All Rights Reserved.

import argparse
import os
import subprocess
from datetime import datetime
from pathlib import Path

import cv2
import librosa
import torch
from PIL import Image
from transformers import Wav2Vec2Model, Wav2Vec2Processor

from diffsynth import ModelManager, WanVideoPipeline
from model import FantasyTalkingAudioConditionModel
from utils import get_audio_features, resize_image_by_longest_edge, save_video


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")

    parser.add_argument(
        "--wan_model_dir",
        type=str,
        default="./models/Wan2.1-I2V-14B-720P",
        required=False,
        help="The dir of the Wan I2V 14B model.",
    )
    parser.add_argument(
        "--fantasytalking_model_path",
        type=str,
        default="./models/fantasytalking_model.ckpt",
        required=False,
        help="The .ckpt path of fantasytalking model.",
    )
    parser.add_argument(
        "--wav2vec_model_dir",
        type=str,
        default="./models/wav2vec2-base-960h",
        required=False,
        help="The dir of wav2vec model.",
    )

    parser.add_argument(
        "--image_path",
        type=str,
        default="./assets/images/woman.png",
        required=False,
        help="The path of the image.",
    )

    parser.add_argument(
        "--audio_path",
        type=str,
        default="./assets/audios/woman.wav",
        required=False,
        help="The path of the audio.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="A woman is talking.",
        required=False,
        help="prompt.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./output",
        help="Dir to save the model.",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=512,
        help="The image will be resized proportionally to this size.",
    )
    parser.add_argument(
        "--audio_scale",
        type=float,
        default=1.0,
        help="Audio condition injection weight",
    )
    parser.add_argument(
        "--prompt_cfg_scale",
        type=float,
        default=5.0,
        required=False,
        help="Prompt cfg scale",
    )
    parser.add_argument(
        "--audio_cfg_scale",
        type=float,
        default=5.0,
        required=False,
        help="Audio cfg scale",
    )
    parser.add_argument(
        "--max_num_frames",
        type=int,
        default=81,
        required=False,
        help="The maximum frames for generating videos, the audio part exceeding max_num_frames/fps will be truncated.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=23,
        required=False,
    )
    parser.add_argument(
        "--num_persistent_param_in_dit",
        type=int,
        default=None,
        required=False,
        help="Maximum parameter quantity retained in video memory, small number to reduce VRAM required",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1111,
        required=False,
    )
    args = parser.parse_args()
    return args


def load_models(args):
    # Load Wan I2V models
    model_manager = ModelManager(device="cpu")
    model_manager.load_models(
        [
            [
                f"{args.wan_model_dir}/diffusion_pytorch_model-00001-of-00007.safetensors",
                f"{args.wan_model_dir}/diffusion_pytorch_model-00002-of-00007.safetensors",
                f"{args.wan_model_dir}/diffusion_pytorch_model-00003-of-00007.safetensors",
                f"{args.wan_model_dir}/diffusion_pytorch_model-00004-of-00007.safetensors",
                f"{args.wan_model_dir}/diffusion_pytorch_model-00005-of-00007.safetensors",
                f"{args.wan_model_dir}/diffusion_pytorch_model-00006-of-00007.safetensors",
                f"{args.wan_model_dir}/diffusion_pytorch_model-00007-of-00007.safetensors",
            ],
            f"{args.wan_model_dir}/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth",
            f"{args.wan_model_dir}/models_t5_umt5-xxl-enc-bf16.pth",
            f"{args.wan_model_dir}/Wan2.1_VAE.pth",
        ],
        # torch_dtype=torch.float8_e4m3fn, # You can set `torch_dtype=torch.bfloat16` to disable FP8 quantization.
        torch_dtype=torch.bfloat16,  # You can set `torch_dtype=torch.bfloat16` to disable FP8 quantization.
    )
    pipe = WanVideoPipeline.from_model_manager(
        model_manager, torch_dtype=torch.bfloat16, device="cuda"
    )

    # Load FantasyTalking weights
    fantasytalking = FantasyTalkingAudioConditionModel(pipe.dit, 768, 2048).to("cuda")
    fantasytalking.load_audio_processor(args.fantasytalking_model_path, pipe.dit)

    # You can set `num_persistent_param_in_dit` to a small number to reduce VRAM required.
    pipe.enable_vram_management(
        num_persistent_param_in_dit=args.num_persistent_param_in_dit
    )

    # Load wav2vec models
    wav2vec_processor = Wav2Vec2Processor.from_pretrained(args.wav2vec_model_dir)
    wav2vec = Wav2Vec2Model.from_pretrained(args.wav2vec_model_dir).to("cuda")

    return pipe, fantasytalking, wav2vec_processor, wav2vec


def main(args, pipe, fantasytalking, wav2vec_processor, wav2vec):
    os.makedirs(args.output_dir, exist_ok=True)

    duration = librosa.get_duration(filename=args.audio_path)  # 计算音频时长
    num_frames = min(int(args.fps * duration // 4) * 4 + 5, args.max_num_frames)

    audio_wav2vec_fea = get_audio_features(
        wav2vec, wav2vec_processor, args.audio_path, args.fps, num_frames
    )  # torch.Size([1, 175, 768])
    image = resize_image_by_longest_edge(args.image_path, args.image_size)  # args.image_size: 512
    width, height = image.size

    audio_proj_fea = fantasytalking.get_proj_fea(audio_wav2vec_fea)  # [1, 175, 768] -> [1, 175, 2048]
    pos_idx_ranges = fantasytalking.split_audio_sequence(
        audio_proj_fea.size(1), num_frames=num_frames
    )  # 21 是音频分割的数量对应于latent frames,21个起始坐标和终止坐标
    audio_proj_split, audio_context_lens = fantasytalking.split_tensor_with_padding(
        audio_proj_fea, pos_idx_ranges, expand_length=4
    )  # 本函数将音频投影特征分割成多个片段，并返回每个片段的有效长度，expand_length=4将实际长度扩展到左边界是4,右边界是4 audio_proj_split: [1, 21, 17, 2048], 21是音频分割的数量对应于latent frames,17是每个片段的长度(左边界和右边界扩展后的结果)，2048是投影特征的维度 audio_context_lens [21]

    # Image-to-video
    video_audio = pipe(
        prompt=args.prompt,
        negative_prompt="人物静止不动，静止，色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
        input_image=image,
        width=width,
        height=height,
        num_frames=num_frames,
        num_inference_steps=30,
        seed=args.seed,
        tiled=True,
        audio_scale=args.audio_scale,
        cfg_scale=args.prompt_cfg_scale,
        audio_cfg_scale=args.audio_cfg_scale,
        audio_proj=audio_proj_split,
        audio_context_lens=audio_context_lens,
        latents_num_frames=(num_frames - 1) // 4 + 1,
    )
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path_tmp = f"{args.output_dir}/tmp_{Path(args.image_path).stem}_{Path(args.audio_path).stem}_{current_time}.mp4"
    save_video(video_audio, save_path_tmp, fps=args.fps, quality=5)

    save_path = f"{args.output_dir}/{Path(args.image_path).stem}_{Path(args.audio_path).stem}_{current_time}.mp4"
    final_command = [
        "ffmpeg",
        "-y",
        "-i",
        save_path_tmp,
        "-i",
        args.audio_path,
        "-c:v",
        "libx264",
        "-c:a",
        "aac",
        "-shortest",
        save_path,
    ]
    subprocess.run(final_command, check=True)
    os.remove(save_path_tmp)
    return save_path


if __name__ == "__main__":
    args = parse_args()
    pipe, fantasytalking, wav2vec_processor, wav2vec = load_models(args)

    main(args, pipe, fantasytalking, wav2vec_processor, wav2vec)
