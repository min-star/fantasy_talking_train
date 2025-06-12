# Copyright Alibaba Inc. All Rights Reserved.

import argparse
from datetime import datetime
from pathlib import Path

import gradio as gr
import librosa

from infer import load_models, main

pipe, fantasytalking, wav2vec_processor, wav2vec = None, None, None, None


def generate_video(
    image_path,
    audio_path,
    prompt,
    prompt_cfg_scale,
    audio_cfg_scale,
    audio_weight,
    image_size,
    max_num_frames,
    inference_steps,
    seed,
):
    # Create the temp directory if it doesn't exist
    output_dir = Path("./output")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Convert paths to absolute Path objects and normalize them
    print(image_path)
    image_path = Path(image_path).absolute().as_posix()
    audio_path = Path(audio_path).absolute().as_posix()

    # Parse the arguments

    args = create_args(
        image_path=image_path,
        audio_path=audio_path,
        prompt=prompt,
        output_dir=str(output_dir),
        audio_weight=audio_weight,
        prompt_cfg_scale=prompt_cfg_scale,
        audio_cfg_scale=audio_cfg_scale,
        image_size=image_size,
        max_num_frames=max_num_frames,
        inference_steps=inference_steps,
        seed=seed,
    )

    try:
        global pipe, fantasytalking, wav2vec_processor, wav2vec
        if pipe is None:
            pipe, fantasytalking, wav2vec_processor, wav2vec = load_models(args)
        output_path = main(args, pipe, fantasytalking, wav2vec_processor, wav2vec)
        return output_path  # Ensure the output path is returned
    except Exception as e:
        print(f"Error during processing: {str(e)}")
        raise gr.Error(f"Error during processing: {str(e)}")


def create_args(
    image_path: str,
    audio_path: str,
    prompt: str,
    output_dir: str,
    audio_weight: float,
    prompt_cfg_scale: float,
    audio_cfg_scale: float,
    image_size: int,
    max_num_frames: int,
    inference_steps: int,
    seed: int,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
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
        help="Dir to save the video.",
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
        help="Image width.",
    )
    parser.add_argument(
        "--prompt_cfg_scale",
        type=float,
        default=5.0,
        required=False,
        help="prompt cfg scale",
    )
    parser.add_argument(
        "--audio_cfg_scale",
        type=float,
        default=5.0,
        required=False,
        help="audio cfg scale",
    )
    parser.add_argument(
        "--max_num_frames",
        type=int,
        default=81,
        required=False,
        help="The maximum frames for generating videos, the audio part exceeding max_num_frames/fps will be truncated.",
    )
    parser.add_argument(
        "--inference_steps",
        type=int,
        default=20,
        required=False,
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
    args = parser.parse_args(
        [
            "--image_path",
            image_path,
            "--audio_path",
            audio_path,
            "--prompt",
            prompt,
            "--output_dir",
            output_dir,
            "--image_size",
            str(image_size),
            "--audio_scale",
            str(audio_weight),
            "--prompt_cfg_scale",
            str(prompt_cfg_scale),
            "--audio_cfg_scale",
            str(audio_cfg_scale),
            "--max_num_frames",
            str(max_num_frames),
            "--inference_steps",
            str(inference_steps),
            "--seed",
            str(seed),
        ]
    )
    print(args)
    return args


# Create Gradio interface
with gr.Blocks(title="FantasyTalking Video Generation") as demo:
    gr.Markdown(
        """
    # FantasyTalking: Realistic Talking Portrait Generation via Coherent Motion Synthesis

    <div align="center">
        <strong> Mengchao Wang1*  Qiang Wang1*  Fan Jiang1†
        Yaqi Fan2    Yunpeng Zhang1,2   YongGang Qi2‡
        Kun Zhao1.   Mu Xu1 </strong>
    </div>

    <div align="center">
        <strong>1AMAP,Alibaba Group   2Beijing University of Posts and Telecommunications</strong>
    </div>

    <div style="display:flex;justify-content:center;column-gap:4px;">
        <a href="https://github.com/Fantasy-AMAP/fantasy-talking">
            <img src='https://img.shields.io/badge/GitHub-Repo-blue'>
        </a>
        <a href="https://arxiv.org/abs/2504.04842">
            <img src='https://img.shields.io/badge/ArXiv-Paper-red'>
        </a>
    </div>
    """
    )

    with gr.Row():
        with gr.Column():
            image_input = gr.Image(label="Input Image", type="filepath")
            audio_input = gr.Audio(label="Input Audio", type="filepath")
            prompt_input = gr.Text(label="Input Prompt")
            with gr.Row():
                prompt_cfg_scale = gr.Slider(
                    minimum=1.0,
                    maximum=9.0,
                    value=5.0,
                    step=0.5,
                    label="Prompt CFG Scale",
                )
                audio_cfg_scale = gr.Slider(
                    minimum=1.0,
                    maximum=9.0,
                    value=5.0,
                    step=0.5,
                    label="Audio CFG Scale",
                )
                audio_weight = gr.Slider(
                    minimum=0.1,
                    maximum=3.0,
                    value=1.0,
                    step=0.1,
                    label="Audio Weight",
                )
            with gr.Row():
                image_size = gr.Number(
                    value=512, label="Width/Height Maxsize", precision=0
                )
                max_num_frames = gr.Number(
                    value=81, label="The Maximum Frames", precision=0
                )
                inference_steps = gr.Slider(
                    minimum=1, maximum=50, value=20, step=1, label="Inference Steps"
                )

            with gr.Row():
                seed = gr.Number(value=1247, label="Random Seed", precision=0)

            process_btn = gr.Button("Generate Video")

        with gr.Column():
            video_output = gr.Video(label="Output Video")

            gr.Examples(
                examples=[
                    [
                        "assets/images/woman.png",
                        "assets/audios/woman.wav",
                    ],
                ],
                inputs=[image_input, audio_input],
            )

    process_btn.click(
        fn=generate_video,
        inputs=[
            image_input,
            audio_input,
            prompt_input,
            prompt_cfg_scale,
            audio_cfg_scale,
            audio_weight,
            image_size,
            max_num_frames,
            inference_steps,
            seed,
        ],
        outputs=video_output,
    )

if __name__ == "__main__":
    demo.launch(inbrowser=True, share=True)
