
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

from transformers import ClapTextModelWithProjection, AutoTokenizer

from train_decoder import create_dataloaders, get_example_data
from dalle2_pytorch import DiffusionPriorNetwork, DiffusionPrior
from dalle2_pytorch.train_configs import TrainDiffusionPriorConfig

from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel

from meldataset import MelDataset, mel_spectrogram, get_dataset_filelist, spectral_normalize_torch, spectral_de_normalize_torch

# riffusion repo must be script working directory

name = "laion/clap-htsat-unfused"

device = 'cuda' if torch.cuda.is_available() else 'cpu'

audioLDMpipe = AudioLDMPipeline.from_pretrained( "cvssp/audioldm-s-full-v2", local_files_only=True )
audioLDMpipe.to(device)

decoder_base_path = '.decoder_only_inference'

if not os.path.exists(decoder_base_path):
    os.mkdir(decoder_base_path)

if not os.path.exists(decoder_base_path + '/decoder_inference'):
    os.mkdir(decoder_base_path + '/decoder_inference')

decoder_config_path = 'configs/train_decoder_config.audio.decoder_only_inference.json'

from train_decoder import create_tracker, recall_trainer, generate_samples
from dalle2_pytorch.trainer import DecoderTrainer
from accelerate import Accelerator

from torchvision.transforms import Resize

accelerator = Accelerator()
config = TrainDecoderConfig.from_json_path(str(decoder_config_path))
tracker = create_tracker(accelerator, config, decoder_config_path, dummy=False)

decoder = config.decoder.create()

trainer = DecoderTrainer(
    decoder=decoder,
    accelerator=accelerator,
)
trainer.to(device)

if tracker.can_recall:
    recall_trainer(tracker, trainer)
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


examples = get_example_data(dataloaders['train'], device, 10)

clip = None
start_unet = 0
end_unet = 3
condition_on_text_encodings = False # todo is it?
cond_scale = 1.0
text_prepend = ""

real_images, generated_images, captions = generate_samples(trainer, examples, clip, start_unet, end_unet, condition_on_text_encodings, cond_scale, device, text_prepend)

from inference_utils import save_melspec

for real_image, generated_image, input_text in zip(real_images, generated_images, captions):

    print("generated_image.shape ", generated_image.shape)
    print("real_image.shape      ", real_image.shape)

    gen_melspec_t = spectral_normalize_torch(generated_image).detach()
    target_melspec_t = spectral_normalize_torch(real_image).detach()

    save_melspec(audioLDMpipe, decoder_base_path, gen_melspec_t, input_text, melspec_type="gen")
    save_melspec(audioLDMpipe, decoder_base_path, target_melspec_t, input_text, melspec_type="tgt")


print("done")
