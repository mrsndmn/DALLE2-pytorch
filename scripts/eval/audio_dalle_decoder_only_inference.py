
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

from meldataset import MelDataset, mel_spectrogram, get_dataset_filelist, spectral_normalize_torch, spectral_de_normalize_torch

from audio_dalle_full_inference import make_inference_config

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

decoder_config_path = 'configs/train_decoder_config.audio.full_no_prior_A100_u0.json'

_, decoder_inference_config_path = make_inference_config(decoder_config_path)

from train_decoder import create_tracker, recall_trainer, generate_samples
from dalle2_pytorch.trainer import DecoderTrainer
from accelerate import Accelerator

from torchvision.transforms import Resize

accelerator = Accelerator()
config = TrainDecoderConfig.from_json_path(str(decoder_inference_config_path))

decoder = config.decoder.create()

trainer = DecoderTrainer(
    decoder=decoder,
    accelerator=accelerator,
)
trainer.to(device)
trainer.eval()

tracker = create_tracker(accelerator, config, decoder_inference_config_path, dummy=False)

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
start_unet = 1
end_unet = 1
condition_on_text_encodings = False # todo is it?
cond_scale = 1.0
text_prepend = ""

real_images, generated_images, captions, youtube_ids = generate_samples(trainer, examples, clip, start_unet, end_unet, condition_on_text_encodings, cond_scale, device, text_prepend, match_image_size=True)

from scripts.eval.inference_utils import save_melspec

for real_image, generated_image, input_text, youtube_id in zip(real_images, generated_images, captions, youtube_ids):

    print("generated_image.shape ", generated_image.shape)
    print("real_image.shape      ", real_image.shape)

    gen_melspec_t = spectral_normalize_torch(generated_image).detach()
    target_melspec_t = spectral_normalize_torch(real_image).detach()

    save_melspec(audioLDMpipe, decoder_base_path, gen_melspec_t, input_text, file_prefix=youtube_id, melspec_type="0gen")
    save_melspec(audioLDMpipe, decoder_base_path, target_melspec_t, input_text, file_prefix=youtube_id, melspec_type="0tgt")


real_images, img_embeddings, text_embeddings, txts, youtube_ids = zip(*examples)
examples = list(zip(generated_images, img_embeddings, text_embeddings, txts, youtube_ids))

start_unet = 1
end_unet = 1

decoder_config_path = 'configs/train_decoder_config.audio.full_no_prior_A100_u1.json'
_, decoder_inference_config_path = make_inference_config(decoder_config_path)

config = TrainDecoderConfig.from_json_path(str(decoder_inference_config_path))

decoder = config.decoder.create()

trainer = DecoderTrainer(
    decoder=decoder,
    accelerator=accelerator,
)
trainer.to(device)
trainer.eval()

tracker = create_tracker(accelerator, config, decoder_inference_config_path, dummy=False)

if tracker.can_recall:
    recall_trainer(tracker, trainer)
else:
    raise Exception("\n\n\n!!!!NO RECALL WAS CALLED!!!!\n\n\n")


real_images, generated_images, captions, youtube_ids = generate_samples(trainer, examples, clip, start_unet, end_unet, condition_on_text_encodings, cond_scale, device, text_prepend, match_image_size=True)

for real_image, generated_image, input_text, youtube_id in zip(real_images, generated_images, captions, youtube_ids):

    print("generated_image.shape ", generated_image.shape)
    print("real_image.shape      ", real_image.shape)

    gen_melspec_t = spectral_normalize_torch(generated_image).detach()
    target_melspec_t = spectral_normalize_torch(real_image).detach()

    save_melspec(audioLDMpipe, decoder_base_path, gen_melspec_t, input_text, file_prefix=youtube_id, melspec_type="1gen")
    save_melspec(audioLDMpipe, decoder_base_path, target_melspec_t, input_text, file_prefix=youtube_id, melspec_type="1tgt")


real_images, img_embeddings, text_embeddings, txts, youtube_ids = zip(*examples)
examples = list(zip(generated_images, img_embeddings, text_embeddings, txts, youtube_ids))

start_unet = 2
end_unet = 2

decoder_config_path = 'configs/train_decoder_config.audio.full_no_prior_A100_u2.json'
_, decoder_inference_config_path = make_inference_config(decoder_config_path)

config = TrainDecoderConfig.from_json_path(str(decoder_config_path))

decoder = config.decoder.create()

trainer = DecoderTrainer(
    decoder=decoder,
    accelerator=accelerator,
)
trainer.to(device)
trainer.eval()

tracker = create_tracker(accelerator, config, decoder_inference_config_path, dummy=False)

if tracker.can_recall:
    recall_trainer(tracker, trainer)
else:
    raise Exception("\n\n\n!!!!NO RECALL WAS CALLED!!!!\n\n\n")

real_images, generated_images, captions, youtube_ids = generate_samples(trainer, examples, clip, start_unet, end_unet, condition_on_text_encodings, cond_scale, device, text_prepend, match_image_size=True)

for real_image, generated_image, input_text, youtube_id in zip(real_images, generated_images, captions, youtube_ids):

    print("generated_image.shape ", generated_image.shape)
    print("real_image.shape      ", real_image.shape)

    gen_melspec_t = spectral_normalize_torch(generated_image).detach()
    target_melspec_t = spectral_normalize_torch(real_image).detach()

    save_melspec(audioLDMpipe, decoder_base_path, gen_melspec_t, input_text, file_prefix=youtube_id, melspec_type="2gen")
    save_melspec(audioLDMpipe, decoder_base_path, target_melspec_t, input_text, file_prefix=youtube_id, melspec_type="2tgt")


print("done")
