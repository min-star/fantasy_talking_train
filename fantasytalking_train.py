import torch, os, imageio, argparse
from torchvision.transforms import v2
from einops import rearrange
import lightning as pl
import pandas as pd
from diffsynth import WanVideoPipeline, ModelManager, load_state_dict
from peft import LoraConfig, inject_adapter_in_model
import torchvision
from PIL import Image
import numpy as np


from transformers import Wav2Vec2Model, Wav2Vec2Processor
import librosa
from model import FantasyTalkingAudioConditionModel
from utils import get_audio_features, resize_image_by_longest_edge, save_video


class TextVideoDataset(torch.utils.data.Dataset):
    def __init__(self, base_path, metadata_path, max_num_frames=81, frame_interval=1, num_frames=81, height=480, width=832, is_i2v=False):
        metadata = pd.read_csv(metadata_path)
        self.path = [os.path.join(base_path, "train", file_name) for file_name in metadata["file_name"]]
        self.text = metadata["text"].to_list()
        self.audio_path = [os.path.join(base_path, "train", file_name.replace(".mp4", ".wav")) for file_name in metadata["audio"]]
        
        self.max_num_frames = max_num_frames
        self.frame_interval = frame_interval
        self.num_frames = num_frames
        self.height = height
        self.width = width
        self.is_i2v = is_i2v
            
        self.frame_process = v2.Compose([
            v2.CenterCrop(size=(height, width)),
            v2.Resize(size=(height, width), antialias=True),
            v2.ToTensor(),
            v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

        
        
    def crop_and_resize(self, image):
        width, height = image.size
        scale = max(self.width / width, self.height / height)
        image = torchvision.transforms.functional.resize(
            image,
            (round(height*scale), round(width*scale)),
            interpolation=torchvision.transforms.InterpolationMode.BILINEAR
        )
        return image


    def load_frames_using_imageio(self, file_path, max_num_frames, start_frame_id, interval, num_frames, frame_process):
        reader = imageio.get_reader(file_path)
        if reader.count_frames() < max_num_frames or reader.count_frames() - 1 < start_frame_id + (num_frames - 1) * interval:
            reader.close()
            return None
        
        frames = []
        first_frame = None
        for frame_id in range(num_frames):
            frame = reader.get_data(start_frame_id + frame_id * interval)
            frame = Image.fromarray(frame)
            frame = self.crop_and_resize(frame)
            if first_frame is None:
                first_frame = frame
            frame = frame_process(frame)
            frames.append(frame)
        reader.close()

        frames = torch.stack(frames, dim=0)
        frames = rearrange(frames, "T C H W -> C T H W")
        
        first_frame = v2.functional.center_crop(first_frame, output_size=(self.height, self.width))
        first_frame = np.array(first_frame)

        if self.is_i2v:
            return frames, first_frame
        else:
            return frames


    def load_video(self, file_path):
        start_frame_id = torch.randint(0, self.max_num_frames - (self.num_frames - 1) * self.frame_interval, (1,))[0]
        frames = self.load_frames_using_imageio(file_path, self.max_num_frames, start_frame_id, self.frame_interval, self.num_frames, self.frame_process)
        return frames
    
    
    def is_image(self, file_path):
        file_ext_name = file_path.split(".")[-1]
        if file_ext_name.lower() in ["jpg", "jpeg", "png", "webp"]:
            return True
        return False
    
    
    def load_image(self, file_path):
        frame = Image.open(file_path).convert("RGB")
        frame = self.crop_and_resize(frame)
        first_frame = frame
        frame = self.frame_process(frame)
        frame = rearrange(frame, "C H W -> C 1 H W")
        return frame
    
    def load_audio(self, file_path):
        #audio
        duration = librosa.get_duration(filename=file_path)  # 计算音频时长
        num_frames = min(int(23 * duration // 4) * 4 + 5, 81)

        audio_wav2vec_fea = get_audio_features(
            self.wav2vec, self.wav2vec_processor, file_path, 23, num_frames
        )  # torch.Size([1, 175, 768])
        return audio_wav2vec_fea


    def __getitem__(self, data_id):
        text = self.text[data_id]
        path = self.path[data_id]
        auido_path = self.audio_path[data_id]
        if self.is_image(path):
            if self.is_i2v:
                raise ValueError(f"{path} is not a video. I2V model doesn't support image-to-image training.")
            video = self.load_image(path)
        else:
            video = self.load_video(path)
        # audio = self.load_audio(auido_path)
        if self.is_i2v:
            video, first_frame = video
            data = {"text": text, "video": video, "path": path, "first_frame": first_frame, "audio_path": auido_path}
        else:
            data = {"text": text, "video": video, "path": path, "audio_path": auido_path}
        return data
    

    def __len__(self):
        return len(self.path)
    

class LightningModelForDataProcess(pl.LightningModule):
    def __init__(self, text_encoder_path, vae_path, image_encoder_path=None, tiled=False, tile_size=(34, 34), tile_stride=(18, 16)):
        super().__init__()
        dit_path = "models/Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00001-of-00007.safetensors,models/Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00002-of-00007.safetensors,models/Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00003-of-00007.safetensors,models/Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00004-of-00007.safetensors,models/Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00005-of-00007.safetensors,models/Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00006-of-00007.safetensors,models/Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00007-of-00007.safetensors"
        model_path = [text_encoder_path, vae_path]
        if image_encoder_path is not None:
            model_path.append(image_encoder_path)
        model_manager = ModelManager(torch_dtype=torch.bfloat16, device="cpu")
        model_manager.load_models(model_path)
        if os.path.isfile(dit_path):
            model_manager.load_models([dit_path])
        else:
            dit_path = dit_path.split(",")
            model_manager.load_models([dit_path])

        self.pipe = WanVideoPipeline.from_model_manager(model_manager)


        self.fantasytalking = FantasyTalkingAudioConditionModel(self.pipe.dit, 768, 2048).to("cuda")
        self.wav2vec = Wav2Vec2Model.from_pretrained("./models/wav2vec2-base-960h").to("cuda")
        self.wav2vec_processor = Wav2Vec2Processor.from_pretrained("./models/wav2vec2-base-960h")

        self.tiler_kwargs = {"tiled": tiled, "tile_size": tile_size, "tile_stride": tile_stride}
        
    def test_step(self, batch, batch_idx):
        text, video, path, audio_path = batch["text"][0], batch["video"], batch["path"][0], batch["audio_path"][0]
        
        self.pipe.device = self.device
        if video is not None:
            # prompt
            prompt_emb = self.pipe.encode_prompt(text)
            # video
            video = video.to(dtype=self.pipe.torch_dtype, device=self.pipe.device)
            latents = self.pipe.encode_video(video, **self.tiler_kwargs)[0]
            # image
            if "first_frame" in batch:
                first_frame = Image.fromarray(batch["first_frame"][0].cpu().numpy())
                _, _, num_frames, height, width = video.shape
                image_emb = self.pipe.encode_image(first_frame, num_frames, height, width)
            else:
                image_emb = {}

            #audio
            duration = librosa.get_duration(filename=audio_path)  # 计算音频时长
            num_frames = min(int(23 * duration // 4) * 4 + 5, 81)

            audio_wav2vec_fea = get_audio_features(
                self.wav2vec, self.wav2vec_processor, audio_path, 23, num_frames
            )  # torch.Size([1, 175, 768])
            data = {"latents": latents, "prompt_emb": prompt_emb, "image_emb": image_emb, "audio_wav2vec_fea": audio_wav2vec_fea}
            torch.save(data, path + ".tensors.pth")  # latents:[16,21,60,104]  prompts:[18, 4096] image_emb:{"clip_fea":[1, 257, 1280], "y":[20,21,60,104]}



class TensorDataset(torch.utils.data.Dataset):
    def __init__(self, base_path, metadata_path, steps_per_epoch):
        metadata = pd.read_csv(metadata_path)
        self.path = [os.path.join(base_path, "train", file_name) for file_name in metadata["file_name"]]
        print(len(self.path), "videos in metadata.")
        self.path = [i + ".tensors.pth" for i in self.path if os.path.exists(i + ".tensors.pth")]
        print(len(self.path), "tensors cached in metadata.")
        assert len(self.path) > 0
        
        self.steps_per_epoch = steps_per_epoch


    def __getitem__(self, index):
        data_id = torch.randint(0, len(self.path), (1,))[0]
        data_id = (data_id + index) % len(self.path) # For fixed seed.
        path = self.path[data_id]
        data = torch.load(path, weights_only=True, map_location="cpu")
        return data
    

    def __len__(self):
        return self.steps_per_epoch


class LightningModelForTrain(pl.LightningModule):
    def __init__(
        self,
        dit_path,
        learning_rate=1e-5,
        use_gradient_checkpointing=True, use_gradient_checkpointing_offload=False,

    ):
        super().__init__()
        model_manager = ModelManager(torch_dtype=torch.bfloat16, device="cpu")
        if os.path.isfile(dit_path):
            model_manager.load_models([dit_path])
        else:
            dit_path = dit_path.split(",")
            model_manager.load_models([dit_path])
        
        self.pipe = WanVideoPipeline.from_model_manager(model_manager)
        self.pipe.scheduler.set_timesteps(1000, training=True)
        # self.pipe.dit.to(dtype=torch.bfloat16)
        self.freeze_parameters()
        
         # Load FantasyTalking weights
        self.fantasytalking = FantasyTalkingAudioConditionModel(self.pipe.dit, 768, 2048).to("cuda")
        # fantasytalking.load_audio_processor(args.fantasytalking_model_path, pipe.dit)
        

        # trainable_param_names = list(filter(lambda named_param: named_param[1].requires_grad, self.pipe.denoising_model().named_parameters()))
        # trainable_param_names_ = list(filter(lambda named_param: named_param[1].requires_grad, self.fantasytalking.named_parameters()))

        
        
        self.learning_rate = learning_rate
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.use_gradient_checkpointing_offload = use_gradient_checkpointing_offload



    def freeze_parameters(self):
        # Freeze parameters
        self.pipe.requires_grad_(False)
        self.pipe.eval()
        self.pipe.denoising_model().train()
        # self.fantasytalking.proj_model.train()

    
    def training_step(self, batch, batch_idx):
        # Data
        latents = batch["latents"].to(self.device)  # torch.Size([1, 16, 21, 60, 104])
        prompt_emb = batch["prompt_emb"]  # 
        prompt_emb["context"] = prompt_emb["context"][0].to(self.device)  # prompt_emb["context"][0]:torch.Size([1, 21, 4096])
        image_emb = batch["image_emb"]
        if "clip_fea" in image_emb:
            image_emb["clip_fea"] = image_emb["clip_fea"][0].to(self.device)  # image_emb["clip_feature"][0]:torch.Size([1, 257, 1280])
        if "y" in image_emb:
            image_emb["y"] = image_emb["y"][0].to(self.device)  # image_emb["y"][0]:torch.Size([1, 21, 60, 104])

        #audio的处理
        audio_wav2vec_fea = batch["audio_wav2vec_fea"].to(self.device)
        #去掉维度为1的第一个维度
        audio_wav2vec_fea = audio_wav2vec_fea.squeeze(1)  # torch.Size([1, 150, 768])
        audio_proj_fea = self.fantasytalking.get_proj_fea(audio_wav2vec_fea)  # torch.Size([1, 150, 768]) -> torch.Size([1, 150, 2048])
        pos_idx_ranges = self.fantasytalking.split_audio_sequence(
        audio_proj_fea.size(1), num_frames=81)   # 21 是音频分割的数量对应于latent frames,21个起始坐标和终止坐标
        audio_proj_split, audio_context_lens = self.fantasytalking.split_tensor_with_padding(
        audio_proj_fea, pos_idx_ranges, expand_length=4)  # 本函数将音频投影特征分割成多个片段，并返回每个片段的有效长度，expand_length=4将实际长度扩展到左边界是4,右边界是4 audio_proj_split: [1, 21, 17, 2048], 21是音频分割的数量对应于latent frames,17是每个片段的长度(左边界和右边界扩展后的结果)，2048是投影特征的维度 audio_context_lens [21]


        # Loss
        self.pipe.device = self.device
        noise = torch.randn_like(latents)
        timestep_id = torch.randint(0, self.pipe.scheduler.num_train_timesteps, (1,))
        timestep = self.pipe.scheduler.timesteps[timestep_id].to(dtype=self.pipe.torch_dtype, device=self.pipe.device)
        extra_input = self.pipe.prepare_extra_input(latents)
        noisy_latents = self.pipe.scheduler.add_noise(latents, noise, timestep)
        training_target = self.pipe.scheduler.training_target(latents, noise, timestep)

        # Compute loss
        noise_pred = self.pipe.denoising_model()(
            noisy_latents, timestep=timestep, **prompt_emb, **extra_input, **image_emb,
            audio_proj=audio_proj_split, audio_context_len=audio_context_lens,latents_num_frames=21,audio_scale=1.0,
            use_gradient_checkpointing=self.use_gradient_checkpointing,
            use_gradient_checkpointing_offload=self.use_gradient_checkpointing_offload
        )
        loss = torch.nn.functional.mse_loss(noise_pred.float(), training_target.float())
        loss = loss * self.pipe.scheduler.training_weight(timestep)

        # Record log
        self.log("train_loss", loss, prog_bar=True)
        return loss
    

    def configure_optimizers(self):
        from itertools import chain
        param = (param for name, param in self.fantasytalking.proj_model.named_parameters())
        trainable_modules1 = filter(lambda p: p.requires_grad, chain(self.pipe.denoising_model().parameters(),param))
        

        # trainable_modules2 = filter(lambda p: p.requires_grad, param)
        # trainable_modules = chain(trainable_modules1, trainable_modules2)
        print("#"* 50)
        print(f"Trainable parameters: {trainable_modules1}")
        print("#"* 50)
        optimizer = torch.optim.AdamW(trainable_modules1, lr=self.learning_rate)
        return optimizer
    

    def on_save_checkpoint(self, checkpoint):
        checkpoint.clear()
        trainable_param_names = list(filter(lambda named_param: named_param[1].requires_grad, self.pipe.denoising_model().named_parameters())) + list(filter(lambda named_param: named_param[1].requires_grad, self.fantasytalking.proj_model.named_parameters()))
        trainable_param_names = set([named_param[0] for named_param in trainable_param_names])
        # state_dict = self.pipe.denoising_model().state_dict()
        state_dict = self.state_dict()

        new_state_dict = {'.'.join(name.split('.')[2:]): param for name, param in state_dict.items()} # 去掉前两个前缀

        lora_state_dict = {}
        for name, param in new_state_dict.items():
            if name in trainable_param_names:
                lora_state_dict[name] = param
        checkpoint.update(lora_state_dict)
        



def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--task",
        type=str,
        default="data_process",
        required=True,
        choices=["data_process", "train"],
        help="Task. `data_process` or `train`.",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="./data1",
        required=True,
        help="The path of the Dataset.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="./modles",
        help="Path to save the model.",
    )
    parser.add_argument(
        "--text_encoder_path",
        type=str,
        default="./models/Wan2.1-I2V-14B-720P/models_t5_umt5-xxl-enc-bf16.pth",
        help="Path of text encoder.",
    )
    parser.add_argument(
        "--image_encoder_path",
        type=str,
        default=None,
        help="models/Wan2.1-I2V-14B-720P/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth",
    )
    parser.add_argument(
        "--vae_path",
        type=str,
        default="./models/Wan2.1-I2V-14B-720P/Wan2.1_VAE.pth",
        help="Path of VAE.",
    )
    parser.add_argument(
        "--dit_path",
        type=str,
        default=None,
        help="Path of DiT.",
    )
    parser.add_argument(
        "--tiled",
        default=False,
        action="store_true",
        help="Whether enable tile encode in VAE. This option can reduce VRAM required.",
    )
    parser.add_argument(
        "--tile_size_height",
        type=int,
        default=34,
        help="Tile size (height) in VAE.",
    )
    parser.add_argument(
        "--tile_size_width",
        type=int,
        default=34,
        help="Tile size (width) in VAE.",
    )
    parser.add_argument(
        "--tile_stride_height",
        type=int,
        default=18,
        help="Tile stride (height) in VAE.",
    )
    parser.add_argument(
        "--tile_stride_width",
        type=int,
        default=16,
        help="Tile stride (width) in VAE.",
    )
    parser.add_argument(
        "--steps_per_epoch",
        type=int,
        default=500,
        help="Number of steps per epoch.",
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=81,
        help="Number of frames.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=480,
        help="Image height.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=832,
        help="Image width.",
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help="Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-5,
        help="Learning rate.",
    )
    parser.add_argument(
        "--accumulate_grad_batches",
        type=int,
        default=1,
        help="The number of batches in gradient accumulation.",
    )
    parser.add_argument(
        "--max_epochs",
        type=int,
        default=1,
        help="Number of epochs.",
    )
    parser.add_argument(
        "--lora_target_modules",
        type=str,
        default="q,k,v,o,ffn.0,ffn.2",
        help="Layers with LoRA modules.",
    )
    parser.add_argument(
        "--init_lora_weights",
        type=str,
        default="kaiming",
        choices=["gaussian", "kaiming"],
        help="The initializing method of LoRA weight.",
    )
    parser.add_argument(
        "--training_strategy",
        type=str,
        default="auto",
        choices=["auto", "deepspeed_stage_1", "deepspeed_stage_2", "deepspeed_stage_3"],
        help="Training strategy",
    )
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=4,
        help="The dimension of the LoRA update matrices.",
    )
    parser.add_argument(
        "--lora_alpha",
        type=float,
        default=4.0,
        help="The weight of the LoRA update matrices.",
    )
    parser.add_argument(
        "--use_gradient_checkpointing",
        default=False,
        action="store_true",
        help="Whether to use gradient checkpointing.",
    )
    parser.add_argument(
        "--use_gradient_checkpointing_offload",
        default=False,
        action="store_true",
        help="Whether to use gradient checkpointing offload.",
    )
    parser.add_argument(
        "--train_architecture",
        type=str,
        default="lora",
        choices=["lora", "full"],
        help="Model structure to train. LoRA training or full training.",
    )
    parser.add_argument(
        "--pretrained_lora_path",
        type=str,
        default=None,
        help="Pretrained LoRA path. Required if the training is resumed.",
    )
    parser.add_argument(
        "--use_swanlab",
        default=False,
        action="store_true",
        help="Whether to use SwanLab logger.",
    )
    parser.add_argument(
        "--swanlab_mode",
        default=None,
        help="SwanLab mode (cloud or local).",
    )
    args = parser.parse_args()
    return args


def data_process(args):
    dataset = TextVideoDataset(
        args.dataset_path,
        os.path.join(args.dataset_path, "metadata.csv"),
        max_num_frames=args.num_frames,
        frame_interval=1,
        num_frames=args.num_frames,
        height=args.height,
        width=args.width,
        is_i2v=args.image_encoder_path is not None
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        shuffle=False,
        batch_size=1,
        num_workers=args.dataloader_num_workers
    )
    model = LightningModelForDataProcess(
        text_encoder_path=args.text_encoder_path,
        image_encoder_path=args.image_encoder_path,
        vae_path=args.vae_path,
        tiled=args.tiled,
        tile_size=(args.tile_size_height, args.tile_size_width),
        tile_stride=(args.tile_stride_height, args.tile_stride_width),
    )
    trainer = pl.Trainer(
        accelerator="gpu",
        devices="auto",
        default_root_dir=args.output_path,
    )

    trainer.test(model, dataloader)


def train(args):
    dataset = TensorDataset(
        args.dataset_path,
        os.path.join(args.dataset_path, "metadata.csv"),
        steps_per_epoch=args.steps_per_epoch,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        shuffle=True,
        batch_size=1,
        num_workers=args.dataloader_num_workers
    )
    model = LightningModelForTrain(
        dit_path=args.dit_path,
        learning_rate=args.learning_rate,
        # train_architecture=args.train_architecture,
        # lora_rank=args.lora_rank,
        # lora_alpha=args.lora_alpha,
        # lora_target_modules=args.lora_target_modules,
        # init_lora_weights=args.init_lora_weights,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload,
        # pretrained_lora_path=args.pretrained_lora_path,
    )
    if args.use_swanlab:
        from swanlab.integration.pytorch_lightning import SwanLabLogger
        swanlab_config = {"UPPERFRAMEWORK": "DiffSynth-Studio"}
        swanlab_config.update(vars(args))
        swanlab_logger = SwanLabLogger(
            project="wan", 
            name="wan",
            config=swanlab_config,
            mode=args.swanlab_mode,
            logdir=os.path.join(args.output_path, "swanlog"),
        )
        logger = [swanlab_logger]
    else:
        logger = None
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator="gpu",
        devices="auto",
        precision="bf16",
        strategy=args.training_strategy,
        default_root_dir=args.output_path,
        accumulate_grad_batches=args.accumulate_grad_batches,
        callbacks=[pl.pytorch.callbacks.ModelCheckpoint(save_top_k=-1)],
        logger=logger,
    )
    trainer.fit(model, dataloader)



if __name__ == '__main__':
    args = parse_args()
    if args.task == "data_process":
        data_process(args)
    elif args.task == "train":
        train(args)


    




