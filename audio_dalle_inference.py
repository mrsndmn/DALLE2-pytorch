
import json
import pandas as pd
import sys
import os
from tqdm.auto import tqdm

from pathlib import Path
import PIL

from multiprocessing import Pool
import time
import os

import argparse
import numpy as np
import torchaudio
import torch

from transformers import ClapTextModelWithProjection, AutoTokenizer

from dalle2_pytorch import DiffusionPriorNetwork, DiffusionPrior
from dalle2_pytorch.train_configs import TrainDiffusionPriorConfig

from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel

from accelerate import Accelerator
from riffusion.riffusion_pipeline import RiffusionPipeline, preprocess_image

# riffusion repo must be script working directory

# parser = argparse.ArgumentParser(description="Inference for audio dalle.")
# parser.add_argument('--input', type=str, required=True)
# args = parser.parse_args()

# input_text = args.input
input_text = "test"

name = "laion/clap-htsat-unfused"

device = 'cuda' if torch.cuda.is_available() else 'cpu'

print("device", device)

print("getting model")
clap = ClapTextModelWithProjection.from_pretrained(name).to(device).float()
# processor =  ClapProcessor.from_pretrained(name)
print("getting tokenizer")
tokenizer =  AutoTokenizer.from_pretrained(name)


print("run tokenizer prompt", input_text)
processed_inputs_text = tokenizer(text=[ input_text ], padding=True, return_tensors="pt")

print("run clap")
clap_text_outputs = clap(**processed_inputs_text)

print("clap_text_outputs.text_embeds.shape", clap_text_outputs.text_embeds.shape)

prior_train_state = torch.load('.prior/best_checkpoint.pth', map_location=device)

config = TrainDiffusionPriorConfig.from_json_path('configs/train_clap_prior_config.json')

diffusionPrior: DiffusionPrior = config.prior.create()
diffusionPrior.load_state_dict(prior_train_state['model'])

image_embedds = diffusionPrior.p_sample_loop( clap_text_outputs.text_embeds.shape, text_cond = { "text_embed": clap_text_outputs.text_embeds } )

print("image_embedds.shape", image_embedds.shape) # [ 1, 512 ]

# todo make riffusion generation and vocoder

riffusion_checkpoint_path = '/home/dtarasov/workspace/hse-audio-dalle2/riffusion/sd-model-finetuned/checkpoint-45000'

accelerator = Accelerator()

unet = UNet2DConditionModel.from_pretrained(
    'riffusion/riffusion-model-v1', subfolder="unet", revision=None
)

unet = accelerator.prepare(unet)

accelerator.load_state(riffusion_checkpoint_path)


riffusion = RiffusionPipeline.load_checkpoint(riffusion_checkpoint_path, device=device)

generator_start = torch.Generator(device="cpu").manual_seed(42)
generator_end = torch.Generator(device="cpu").manual_seed(43)
generator_latents = torch.Generator(device="cpu").manual_seed(44)

init_image_path = Path("/home/dtarasov/workspace/hse-audio-dalle2/riffusion/seed_images/agile.png")

init_image = PIL.Image.open(str(init_image_path)).convert("RGB")

init_image_torch = preprocess_image(init_image).to(
    device=riffusion.device, dtype=image_embedds.dtype
)
init_image_torch.unsqueeze_(0)

init_latent_dist = riffusion.vae.encode(init_image_torch).latent_dist
init_latents = init_latent_dist.sample(generator=generator_latents)
init_latents = 0.18215 * init_latents

riffusion_generated_spectrogram = riffusion.interpolate_img2img(
    text_embeddings=image_embedds,
    init_latents=init_latents,
    mask=None,
    generator_a=generator_start,
    generator_b=generator_end,
    interpolate_alpha=0.5,
    strength_a=1.0,
    strength_b=1.0,
    # num_inference_steps=50,
    # guidance_scale=7.5,
)
