import os

import numpy as np
import torch
from einops import rearrange
from PIL import Image
from tqdm import tqdm

from ..models import ModelManager
from ..models.wan_video_dit import WanLayerNorm, WanModel, WanRMSNorm
from ..models.wan_video_image_encoder import WanImageEncoder
from ..models.wan_video_text_encoder import (T5LayerNorm, T5RelativeEmbedding,
                                             WanTextEncoder)
from ..models.wan_video_vae import (CausalConv3d, RMS_norm, Upsample,
                                    WanVideoVAE)
from ..prompters import WanPrompter
from ..schedulers.flow_match import FlowMatchScheduler
from ..vram_management import (AutoWrappedLinear, AutoWrappedModule,
                               enable_vram_management)
from .base import BasePipeline


class WanVideoPipeline(BasePipeline):
    def __init__(self, device="cuda", torch_dtype=torch.float16, tokenizer_path=None):
        super().__init__(device=device, torch_dtype=torch_dtype)
        self.scheduler = FlowMatchScheduler(shift=5, sigma_min=0.0, extra_one_step=True)
        self.prompter = WanPrompter(tokenizer_path=tokenizer_path)
        self.text_encoder: WanTextEncoder = None
        self.image_encoder: WanImageEncoder = None
        self.dit: WanModel = None
        self.vae: WanVideoVAE = None
        self.model_names = ["text_encoder", "dit", "vae"]
        self.height_division_factor = 16
        self.width_division_factor = 16

    def enable_vram_management(self, num_persistent_param_in_dit=None):
        dtype = next(iter(self.text_encoder.parameters())).dtype
        enable_vram_management(
            self.text_encoder,
            module_map={
                torch.nn.Linear: AutoWrappedLinear,
                torch.nn.Embedding: AutoWrappedModule,
                T5RelativeEmbedding: AutoWrappedModule,
                T5LayerNorm: AutoWrappedModule,
            },
            module_config=dict(
                offload_dtype=dtype,
                offload_device="cpu",
                onload_dtype=dtype,
                onload_device="cpu",
                computation_dtype=self.torch_dtype,
                computation_device=self.device,
            ),
        )
        dtype = next(iter(self.dit.parameters())).dtype
        enable_vram_management(
            self.dit,
            module_map={
                torch.nn.Linear: AutoWrappedLinear,
                torch.nn.Conv3d: AutoWrappedModule,
                torch.nn.LayerNorm: AutoWrappedModule,
                WanLayerNorm: AutoWrappedModule,
                WanRMSNorm: AutoWrappedModule,
            },
            module_config=dict(
                offload_dtype=dtype,
                offload_device="cpu",
                onload_dtype=dtype,
                onload_device=self.device,
                computation_dtype=self.torch_dtype,
                computation_device=self.device,
            ),
            max_num_param=num_persistent_param_in_dit,
            overflow_module_config=dict(
                offload_dtype=dtype,
                offload_device="cpu",
                onload_dtype=dtype,
                onload_device="cpu",
                computation_dtype=self.torch_dtype,
                computation_device=self.device,
            ),
        )
        dtype = next(iter(self.vae.parameters())).dtype
        enable_vram_management(
            self.vae,
            module_map={
                torch.nn.Linear: AutoWrappedLinear,
                torch.nn.Conv2d: AutoWrappedModule,
                RMS_norm: AutoWrappedModule,
                CausalConv3d: AutoWrappedModule,
                Upsample: AutoWrappedModule,
                torch.nn.SiLU: AutoWrappedModule,
                torch.nn.Dropout: AutoWrappedModule,
            },
            module_config=dict(
                offload_dtype=dtype,
                offload_device="cpu",
                onload_dtype=dtype,
                onload_device=self.device,
                computation_dtype=self.torch_dtype,
                computation_device=self.device,
            ),
        )
        if self.image_encoder is not None:
            dtype = next(iter(self.image_encoder.parameters())).dtype
            enable_vram_management(
                self.image_encoder,
                module_map={
                    torch.nn.Linear: AutoWrappedLinear,
                    torch.nn.Conv2d: AutoWrappedModule,
                    torch.nn.LayerNorm: AutoWrappedModule,
                },
                module_config=dict(
                    offload_dtype=dtype,
                    offload_device="cpu",
                    onload_dtype=dtype,
                    onload_device="cpu",
                    computation_dtype=self.torch_dtype,
                    computation_device=self.device,
                ),
            )
        self.enable_cpu_offload()

    def fetch_models(self, model_manager: ModelManager):
        text_encoder_model_and_path = model_manager.fetch_model(
            "wan_video_text_encoder", require_model_path=True
        )
        if text_encoder_model_and_path is not None:
            self.text_encoder, tokenizer_path = text_encoder_model_and_path
            self.prompter.fetch_models(self.text_encoder)
            self.prompter.fetch_tokenizer(
                os.path.join(os.path.dirname(tokenizer_path), "google/umt5-xxl")
            )
        self.dit = model_manager.fetch_model("wan_video_dit")
        self.vae = model_manager.fetch_model("wan_video_vae")
        self.image_encoder = model_manager.fetch_model("wan_video_image_encoder")

    @staticmethod
    def from_model_manager(model_manager: ModelManager, torch_dtype=None, device=None):
        if device is None:
            device = model_manager.device
        if torch_dtype is None:
            torch_dtype = model_manager.torch_dtype
        pipe = WanVideoPipeline(device=device, torch_dtype=torch_dtype)
        pipe.fetch_models(model_manager)
        return pipe

    def denoising_model(self):
        return self.dit

    def encode_prompt(self, prompt, positive=True):
        prompt_emb = self.prompter.encode_prompt(prompt, positive=positive)
        return {"context": prompt_emb}

    def encode_image(self, image, num_frames, height, width):
        with torch.amp.autocast(
            dtype=torch.bfloat16, device_type=torch.device(self.device).type
        ):
            image = self.preprocess_image(image.resize((width, height))).to(self.device)
            clip_context = self.image_encoder.encode_image([image])  # torch.Size([1, 257, 1280])
            msk = torch.ones(1, num_frames, height // 8, width // 8, device=self.device)  # torch.Size([1, 81, 64, 64])
            msk[:, 1:] = 0
            msk = torch.concat(
                [torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]],
                dim=1,
            )  # torch.Size([1, 84, 64, 64])
            msk = msk.view(1, msk.shape[1] // 4, 4, height // 8, width // 8)  # torch.Size([1, 21, 4, 64, 64])
            msk = msk.transpose(1, 2)[0]  # torch.Size([4, 21, 64, 64])
            y = self.vae.encode(
                [
                    torch.concat(
                        [
                            image.transpose(0, 1),
                            torch.zeros(3, num_frames - 1, height, width).to(
                                image.device
                            ),
                        ],
                        dim=1,
                    )
                ],
                device=self.device,
            )[0]  # torch.Size([16, 21, 64, 64])
            y = torch.concat([msk, y])  # torch.Size([20, 21, 64, 64])
        return {"clip_fea": clip_context, "y": [y]}

    def tensor2video(self, frames):
        frames = rearrange(frames, "C T H W -> T H W C")
        frames = (
            ((frames.float() + 1) * 127.5).clip(0, 255).cpu().numpy().astype(np.uint8)
        )
        frames = [Image.fromarray(frame) for frame in frames]
        return frames

    def prepare_extra_input(self, latents=None):
        return {"seq_len": latents.shape[2] * latents.shape[3] * latents.shape[4] // 4}  # 计算序列长度 f*w*h/(patch_size**2)

    def encode_video(
        self, input_video, tiled=True, tile_size=(34, 34), tile_stride=(18, 16)
    ):
        with torch.amp.autocast(
            dtype=torch.bfloat16, device_type=torch.device(self.device).type
        ):
            latents = self.vae.encode(
                input_video,
                device=self.device,
                tiled=tiled,
                tile_size=tile_size,
                tile_stride=tile_stride,
            )
        return latents

    def decode_video(
        self, latents, tiled=True, tile_size=(34, 34), tile_stride=(18, 16)
    ):
        with torch.amp.autocast(
            dtype=torch.bfloat16, device_type=torch.device(self.device).type
        ):
            frames = self.vae.decode(
                latents,
                device=self.device,
                tiled=tiled,
                tile_size=tile_size,
                tile_stride=tile_stride,
            )
        return frames

    def set_ip(self, local_path):
        pass

    @torch.no_grad()
    def __call__(
        self,
        prompt,
        negative_prompt="",
        input_image=None,
        input_video=None,
        denoising_strength=1.0,
        seed=None,
        rand_device="cpu",
        height=480,
        width=832,
        num_frames=81,
        cfg_scale=5.0,
        audio_cfg_scale=None,
        num_inference_steps=50,
        sigma_shift=5.0,
        tiled=True,
        tile_size=(30, 52),
        tile_stride=(15, 26),
        progress_bar_cmd=tqdm,
        progress_bar_st=None,
        **kwargs,
    ):
        # Parameter check
        height, width = self.check_resize_height_width(height, width)  
        if num_frames % 4 != 1:
            num_frames = (num_frames + 2) // 4 * 4 + 1
            print(
                f"Only `num_frames % 4 != 1` is acceptable. We round it up to {num_frames}."
            )

        # Tiler parameters
        tiler_kwargs = {
            "tiled": tiled,  # True
            "tile_size": tile_size,  # (30, 52)
            "tile_stride": tile_stride,  # (15, 26)
        }

        # Scheduler
        self.scheduler.set_timesteps(
            num_inference_steps, denoising_strength, shift=sigma_shift
        )

        # Initialize noise
        noise = self.generate_noise(
            (1, 16, (num_frames - 1) // 4 + 1, height // 8, width // 8),
            seed=seed,
            device=rand_device,
            dtype=torch.float32,
        ).to(self.device)  # torch.Size([1, 16, 21, 64, 64])
        if input_video is not None:  # 如果输入视频存在，就将其编码为潜在空间，然后添加噪声
            self.load_models_to_device(["vae"])
            input_video = self.preprocess_images(input_video)
            input_video = torch.stack(input_video, dim=2)
            latents = self.encode_video(input_video, **tiler_kwargs).to(
                dtype=noise.dtype, device=noise.device
            )
            latents = self.scheduler.add_noise(
                latents, noise, timestep=self.scheduler.timesteps[0]
            )
        else:
            latents = noise

        # Encode prompts
        self.load_models_to_device(["text_encoder"])
        prompt_emb_posi = self.encode_prompt(prompt, positive=True)  # [15, 4096]
        if cfg_scale != 1.0:
            prompt_emb_nega = self.encode_prompt(negative_prompt, positive=False)  # [135, 4096]

        # Encode image
        if input_image is not None and self.image_encoder is not None:
            self.load_models_to_device(["image_encoder", "vae"])
            image_emb = self.encode_image(input_image, num_frames, height, width)  # clip_fea:torch.Size([1, 257, 1280])  y:[torch.Size([20, 21, 64, 64])] 
        else:
            image_emb = {}

        # Extra input
        extra_input = self.prepare_extra_input(latents)  # {'seq_len': 21504}

        # Denoise
        self.load_models_to_device(["dit"])
        with torch.amp.autocast(
            dtype=torch.bfloat16, device_type=torch.device(self.device).type
        ):
            for progress_id, timestep in enumerate(
                progress_bar_cmd(self.scheduler.timesteps)
            ):
                timestep = timestep.unsqueeze(0).to(
                    dtype=torch.float32, device=self.device
                )

                # Inference
                noise_pred_posi = self.dit(
                    latents,  # torch.Size([1, 16, 21, 64, 64])
                    timestep=timestep,  # tensor([1000.], device='cuda:0')
                    **prompt_emb_posi,  # context: [15, 4096]
                    **image_emb,  # clip_fea:torch.Size([1, 257, 1280])  y:[torch.Size([20, 21, 64, 64])]
                    **extra_input,  # {'seq_len': 21504}
                    **kwargs,  # audio_scale, audio_proj: torch.Size([1, 21, 17, 2048]), audio_context_lens, latents_num_frames
                )  # (zt,audio,prompt)
                if audio_cfg_scale is not None:
                    audio_scale = kwargs["audio_scale"]
                    kwargs["audio_scale"] = 0.0
                    noise_pred_noaudio = self.dit(
                        latents,
                        timestep=timestep,
                        **prompt_emb_posi,
                        **image_emb,
                        **extra_input,
                        **kwargs,
                    )  # (zt,0,prompt)
                    # kwargs['ip_scale'] = ip_scale
                    if cfg_scale != 1.0:  # prompt cfg
                        noise_pred_no_cond = self.dit(
                            latents,
                            timestep=timestep,
                            **prompt_emb_nega,
                            **image_emb,
                            **extra_input,
                            **kwargs,
                        )  # (zt,0,0)
                        noise_pred = (
                            noise_pred_no_cond
                            + cfg_scale * (noise_pred_noaudio - noise_pred_no_cond)
                            + audio_cfg_scale * (noise_pred_posi - noise_pred_noaudio)
                        )
                    else:
                        noise_pred = noise_pred_noaudio + audio_cfg_scale * (
                            noise_pred_posi - noise_pred_noaudio
                        )
                    kwargs["audio_scale"] = audio_scale
                else:
                    if cfg_scale != 1.0:
                        noise_pred_nega = self.dit(
                            latents,
                            timestep=timestep,
                            **prompt_emb_nega,
                            **image_emb,
                            **extra_input,
                            **kwargs,
                        )  # (zt,audio,0)
                        noise_pred = noise_pred_nega + cfg_scale * (
                            noise_pred_posi - noise_pred_nega
                        )
                    else:
                        noise_pred = noise_pred_posi

                # Scheduler
                latents = self.scheduler.step(
                    noise_pred, self.scheduler.timesteps[progress_id], latents
                )

        # Decode
        self.load_models_to_device(["vae"])
        frames = self.decode_video(latents, **tiler_kwargs)
        self.load_models_to_device([])
        frames = self.tensor2video(frames[0])

        return frames
