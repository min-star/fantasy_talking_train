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


def load_models(args):
    # Load Wan model
    wan_model = WanVideoPipeline.from_pretrained(
        args.wan_model_dir, torch_dtype=torch.float16
    )
    wan_model.to("cuda")

    # Load FantasyTalking model
    fantasytalking_model = FantasyTalkingAudioConditionModel.load_from_checkpoint(
        args.fantasytalking_model_path
    )
    fantasytalking_model.to("cuda")

    # Load Wav2Vec2 model
    wav2vec_processor = Wav2Vec2Processor.from_pretrained(args.wav2vec_model_dir)
    wav2vec_model = Wav2Vec2Model.from_pretrained(args.wav2vec_model_dir)
    wav2vec_model.to("cuda")

    return wan_model, fantasytalking_model, wav2vec_processor, wav2vec_model