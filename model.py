import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open

from diffsynth.models.wan_video_dit import WanModel, flash_attention, attention


class AudioProjModel(nn.Module):
    def __init__(self, audio_in_dim=1024, cross_attention_dim=1024):
        super().__init__()
        self.cross_attention_dim = cross_attention_dim
        self.proj = torch.nn.Linear(audio_in_dim, cross_attention_dim, bias=False)  # 1024 -> 2048 通过线性层将音频特征投影到交叉注意力维度
        self.norm = torch.nn.LayerNorm(cross_attention_dim)

    def forward(self, audio_embeds):
        context_tokens = self.proj(audio_embeds)
        context_tokens = self.norm(context_tokens)
        return context_tokens  # [B,L,C]


class WanCrossAttentionProcessor(nn.Module):
    def __init__(self, context_dim, hidden_dim):
        super().__init__()

        self.context_dim = context_dim  # 2048
        self.hidden_dim = hidden_dim  # 5120

        self.k_proj = nn.Linear(context_dim, hidden_dim, bias=False)  # 通过线性映射层得到k,q,k,v的特征维度必须相同 2048 -> 5120
        self.v_proj = nn.Linear(context_dim, hidden_dim, bias=False)  # 通过线性映射层得到v,q,k,v的特征维度必须相同 2048 -> 5120

        nn.init.zeros_(self.k_proj.weight)  # 初始化k_proj的权重为0
        nn.init.zeros_(self.v_proj.weight)  # 初始化v_proj的权重为0

    def __call__(
        self,
        attn: nn.Module,
        x: torch.Tensor,
        context: torch.Tensor,
        context_lens: torch.Tensor,
        audio_proj: torch.Tensor,
        audio_context_lens: torch.Tensor,
        latents_num_frames: int = 21,
        audio_scale: float = 1.0,
    ) -> torch.Tensor:
        """
        x:              [B, L1, C].
        context:        [B, L2, C].
        context_lens:   [B].
        audio_proj:   [B, 21, L3, C]
        audio_context_lens: [B*21].
        """
        context_img = context[:, :257]  # 取出context的前257个token作为图像特征 torch.Size([1, 257, 5120])
        context = context[:, 257:]  # 取出context的后面部分作为文本特征 torch.Size([1, 512, 5120])
        b, n, d = x.size(0), attn.num_heads, attn.head_dim  #1, 40, 128

        # compute query, key, value
        q = attn.norm_q(attn.q(x)).view(b, -1, n, d) # torch.Size([1, 21504, 40, 128])
        k = attn.norm_k(attn.k(context)).view(b, -1, n, d)  # torch.Size([1, 512, 40, 128])
        v = attn.v(context).view(b, -1, n, d)  # torch.Size([1, 512, 40, 128])
        k_img = attn.norm_k_img(attn.k_img(context_img)).view(b, -1, n, d)  # # torch.Size([1, 257, 40, 128])
        v_img = attn.v_img(context_img).view(b, -1, n, d)  # torch.Size([1, 257, 40, 128])
        img_x = flash_attention(q, k_img, v_img, k_lens=None)  # torch.Size([1, 21504, 40, 128])
        # compute attention
        x = flash_attention(q, k, v, k_lens=context_lens)  # torch.Size([1, 21504, 40, 128])
        x = x.flatten(2)  # torch.Size([1, 21504, 5120])
        img_x = img_x.flatten(2)  # torch.Size([1, 21504, 5120])

        if len(audio_proj.shape) == 4:  # torch.Size([1, 21, 17, 2048])
            audio_q = q.view(b * latents_num_frames, -1, n, d)  # torch.Size([21, 1024, 40, 128])[b, 21, l1, n, d]
            ip_key = self.k_proj(audio_proj).view(b * latents_num_frames, -1, n, d)  # torch.Size([21, 17, 40, 128])
            ip_value = self.v_proj(audio_proj).view(b * latents_num_frames, -1, n, d)  # torch.Size([21, 17, 40, 128])
            audio_x = attention(
                audio_q, ip_key, ip_value, k_lens=audio_context_lens
            )  # torch.Size([21, 1024, 40, 128])
            audio_x = audio_x.view(b, q.size(1), n, d)  # torch.Size([1, 21504, 5120])
            audio_x = audio_x.flatten(2)  # torch.Size([1, 21504, 5120])
        elif len(audio_proj.shape) == 3:
            ip_key = self.k_proj(audio_proj).view(b, -1, n, d)
            ip_value = self.v_proj(audio_proj).view(b, -1, n, d)
            audio_x = attention(q, ip_key, ip_value, k_lens=audio_context_lens)
            audio_x = audio_x.flatten(2)
        # output
        x = x + img_x + audio_x * audio_scale  # 将音频特征与图像特征和文本特征进行融合
        x = attn.o(x)
        return x


class FantasyTalkingAudioConditionModel(nn.Module):
    def __init__(self, wan_dit: WanModel, audio_in_dim: int, audio_proj_dim: int):
        super().__init__()

        self.audio_in_dim = audio_in_dim  # 768
        self.audio_proj_dim = audio_proj_dim  # 2048

        # audio proj model
        self.proj_model = self.init_proj(self.audio_proj_dim) 
        self.set_audio_processor(wan_dit)

    def init_proj(self, cross_attention_dim=5120):
        proj_model = AudioProjModel(
            audio_in_dim=self.audio_in_dim, cross_attention_dim=cross_attention_dim
        )
        return proj_model

    def set_audio_processor(self, wan_dit):
        attn_procs = {}  # dict of attention processors
        for name in wan_dit.attn_processors.keys():  # 将现有的注意力处理器替换为WanCrossAttentionProcessor
            attn_procs[name] = WanCrossAttentionProcessor(
                context_dim=self.audio_proj_dim, hidden_dim=wan_dit.dim  # context_dim: 2048, hidden_dim: 5120
            )
        wan_dit.set_attn_processor(attn_procs)

    def load_audio_processor(self, ip_ckpt: str, wan_dit):
        if os.path.splitext(ip_ckpt)[-1] == ".safetensors":
            state_dict = {"proj_model": {}, "audio_processor": {}}
            with safe_open(ip_ckpt, framework="pt", device="cpu") as f:
                for key in f.keys():
                    if key.startswith("proj_model."):
                        state_dict["proj_model"][
                            key.replace("proj_model.", "")
                        ] = f.get_tensor(key)
                    elif key.startswith("audio_processor."):
                        state_dict["audio_processor"][
                            key.replace("audio_processor.", "")
                        ] = f.get_tensor(key)
        else:
            state_dict = torch.load(ip_ckpt, map_location="cpu")
        self.proj_model.load_state_dict(state_dict["proj_model"])  # 将音频投影模型的权重加载到proj_model中
        wan_dit.load_state_dict(state_dict["audio_processor"], strict=False)

    def get_proj_fea(self, audio_fea=None):
        return self.proj_model(audio_fea) if audio_fea is not None else None

    def split_audio_sequence(self, audio_proj_length, num_frames=81):
        """
        Map the audio feature sequence to corresponding latent frame slices.

        Args:
            audio_proj_length (int): The total length of the audio feature sequence
                                    (e.g., 173 in audio_proj[1, 173, 768]).
            num_frames (int): The number of video frames in the training data (default: 81).

        Returns:
            list: A list of [start_idx, end_idx] pairs. Each pair represents the index range
                (within the audio feature sequence) corresponding to a latent frame.
        """
        # Average number of tokens per original video frame
        tokens_per_frame = audio_proj_length / num_frames  # 作用是要将latents和音频进行对齐 2.1604938271604937

        # Each latent frame covers 4 video frames, and we want the center
        tokens_per_latent_frame = tokens_per_frame * 4  # 8.641975308641975
        half_tokens = int(tokens_per_latent_frame / 2)  # 4

        pos_indices = []
        for i in range(int((num_frames - 1) / 4) + 1):
            if i == 0:
                pos_indices.append(0)
            else:
                start_token = tokens_per_frame * ((i - 1) * 4 + 1)
                end_token = tokens_per_frame * (i * 4 + 1)
                center_token = int((start_token + end_token) / 2) - 1
                pos_indices.append(center_token)

        # Build index ranges centered around each position
        pos_idx_ranges = [[idx - half_tokens, idx + half_tokens] for idx in pos_indices]

        # Adjust the first range to avoid negative start index  [-4, 4] -> [-7, 1]
        pos_idx_ranges[0] = [
            -(half_tokens * 2 - pos_idx_ranges[1][0]),
            pos_idx_ranges[1][0],
        ]

        return pos_idx_ranges

    def split_tensor_with_padding(self, input_tensor, pos_idx_ranges, expand_length=0):
        """
        Split the input tensor into subsequences based on index ranges, and apply right-side zero-padding
        if the range exceeds the input boundaries.

        Args:
            input_tensor (Tensor): Input audio tensor of shape [1, L, 768].
            pos_idx_ranges (list): A list of index ranges, e.g. [[-7, 1], [1, 9], ..., [165, 173]].
            expand_length (int): Number of tokens to expand on both sides of each subsequence.

        Returns:
            sub_sequences (Tensor): A tensor of shape [1, F, L, 768], where L is the length after padding.
                                    Each element is a padded subsequence.
            k_lens (Tensor): A tensor of shape [F], representing the actual (unpadded) length of each subsequence.
                            Useful for ignoring padding tokens in attention masks.
        """
        pos_idx_ranges = [
            [idx[0] - expand_length, idx[1] + expand_length] for idx in pos_idx_ranges
        ]
        sub_sequences = []
        seq_len = input_tensor.size(1)  # 175
        max_valid_idx = seq_len - 1  # 174
        k_lens_list = []
        for start, end in pos_idx_ranges:
            # Calculate the fill amount
            pad_front = max(-start, 0)  # How many tokens to pad at the front
            pad_back = max(end - max_valid_idx, 0)  # How many tokens to pad at the back

            # Calculate the start and end indices of the valid part
            valid_start = max(start, 0)
            valid_end = min(end, max_valid_idx)

            # Extract the valid part
            if valid_start <= valid_end:
                valid_part = input_tensor[:, valid_start : valid_end + 1, :]
            else:
                valid_part = input_tensor.new_zeros((1, 0, input_tensor.size(2)))

            # In the sequence dimension (the 1st dimension) perform padding
            padded_subseq = F.pad(
                valid_part,
                (0, 0, 0, pad_back + pad_front, 0, 0),
                mode="constant",
                value=0,
            )
            k_lens_list.append(padded_subseq.size(-2) - pad_back - pad_front)

            sub_sequences.append(padded_subseq)
        return torch.stack(sub_sequences, dim=1), torch.tensor(
            k_lens_list, dtype=torch.long
        )
