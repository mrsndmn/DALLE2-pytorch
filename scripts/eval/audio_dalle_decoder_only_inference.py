
import json
import pandas as pd
import sys
import os
from tqdm.auto import tqdm

import numpy as np

from pathlib import Path
import PIL

from multiprocessing import Pool
import time
import os

import argparse
import numpy as np
import torchaudio
import torch

sys.path.insert(0, '/home/dtarasov/workspace/hse-audio-dalle2/diffusers/src')
sys.path.insert(0, '/home/dtarasov/workspace/hse-audio-dalle2/transformers/src')
sys.path.insert(0, '/home/dtarasov/workspace/hse-audio-dalle2/DALLE2-pytorch')
sys.path.insert(0, '/home/dtarasov/workspace/hse-audio-dalle2/hifi-gan')

from diffusers import AudioLDMPipeline

from dalle2_pytorch.train_configs import DecoderConfig, TrainDecoderConfig

from train_decoder import create_dataloaders, get_example_data
from dalle2_pytorch import DiffusionPriorNetwork, DiffusionPrior
from dalle2_pytorch.train_configs import TrainDiffusionPriorConfig

from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel


from audio_dalle_full_inference import make_inference_config

from scripts.eval.inference_utils import save_melspec

# riffusion repo must be script working directory

name = "laion/clap-htsat-unfused"

device = 'cuda' if torch.cuda.is_available() else 'cpu'

audioLDMpipe = AudioLDMPipeline.from_pretrained( "cvssp/audioldm-s-full-v2", local_files_only=True )
audioLDMpipe.to(device)

inference_out_base_path = '.decoder_only_inference'

if not os.path.exists(inference_out_base_path):
    os.mkdir(inference_out_base_path)

if not os.path.exists(inference_out_base_path + '/decoder_inference'):
    os.mkdir(inference_out_base_path + '/decoder_inference')



from train_decoder import create_tracker, recall_trainer
from dalle2_pytorch.trainer import DecoderTrainer
from accelerate import Accelerator
from dalle2_pytorch.inference import audio_dalle_decoder_generate

from torchvision.transforms import Resize

accelerator = Accelerator()
decoder_config_path = 'configs/inference/train_decoder_config.audio.full_no_prior_A100_u0.json'
# decoder_config_path = 'configs/train_decoder_config.audio.full_no_prior_A100_full.json'
# _, decoder_config_path = make_inference_config('configs/train_decoder_config.audio.full_no_prior_A100_u1.json')

config = TrainDecoderConfig.from_json_path(str(decoder_config_path))

decoder = config.decoder.create()

trainer = DecoderTrainer(
    decoder=decoder,
    accelerator=accelerator,
)
trainer.to(device)
trainer.eval()

tracker = create_tracker(accelerator, config, decoder_config_path, dummy=False)

if tracker.can_recall:
    recall_trainer(tracker, trainer, strict=False)
else:
    raise Exception("\n\n\n!!!!NO RECALL WAS CALLED!!!!\n\n\n")

dataloaders = create_dataloaders(
    available_shards=list(range(config.data.start_shard, config.data.end_shard)),
    audio_preproc = config.data.audio_preproc,
    train_prop = config.data.splits.train,
    val_prop = config.data.splits.val,
    test_prop = config.data.splits.test,
    n_sample_images=config.train.n_sample_images,
    # webdataset_base_url=config.data.webdataset_base_url, # should be in config
    **config.data.dict(),
    seed = config.seed,
)

examples = get_example_data(dataloaders['val'], device, 50)

configs = [
    'configs/inference/train_decoder_config.audio.full_no_prior_A100_u0.json',
    'configs/inference/train_decoder_config.audio.full_no_prior_A100_u1.json',
    'configs/inference/train_decoder_config.audio.full_no_prior_A100_u2.json',
]

audio_dalle_decoder_generate(audioLDMpipe, configs, examples, inference_out_base_path, device=device)

