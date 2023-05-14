
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

from dalle2_pytorch.train_configs import DecoderConfig, TrainDecoderConfig

from transformers import ClapTextModelWithProjection, AutoTokenizer

from train_decoder import create_dataloaders, get_example_data
from dalle2_pytorch import DiffusionPriorNetwork, DiffusionPrior
from dalle2_pytorch.train_configs import TrainDiffusionPriorConfig

from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel

from meldataset import MelDataset, mel_spectrogram, get_dataset_filelist, spectral_normalize_torch, spectral_de_normalize_torch

# riffusion repo must be script working directory

# parser = argparse.ArgumentParser(description="Inference for audio dalle.")
# parser.add_argument('--input', type=str, required=True)
# args = parser.parse_args()

# input_text = args.input

input_text = "A duck quacks"
# input_text = "Dogs bark nearby"
# input_text = "Birds chirping and tweeting"
# input_text = "A device sprays a liquid"
# input_text = "A fly buzzes as the mic picks up wind and birds chirp in the background"

name = "laion/clap-htsat-unfused"

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# do_clap_evaluation = False
do_clap_evaluation = True

print("device", device)

print("getting model")
clap = ClapTextModelWithProjection.from_pretrained(name).float()
# processor =  ClapProcessor.from_pretrained(name)
print("getting tokenizer")
tokenizer =  AutoTokenizer.from_pretrained(name)

print("run tokenizer prompt", input_text)
processed_inputs_text = tokenizer(text=[ input_text ], padding=True, return_tensors="pt")

print("run clap")
clap_text_outputs = clap(**processed_inputs_text)

print("clap_text_outputs.text_embeds.shape", clap_text_outputs.text_embeds.shape) # [ 1, 512 ]

prior_train_state = torch.load('.prior/best_checkpoint.pth', map_location=device)

config = TrainDiffusionPriorConfig.from_json_path('configs/train_clap_prior_config.json')

diffusionPrior: DiffusionPrior = config.prior.create()
diffusionPrior.load_state_dict(prior_train_state['model'])
diffusionPrior.to(device)

clap_text_embeddings_normalized = clap_text_outputs.text_embeds / clap_text_outputs.text_embeds.norm(p=2, dim=-1, keepdim=True)

audio_embedds = diffusionPrior.p_sample_loop( clap_text_outputs.text_embeds.shape, text_cond = { "text_embed": clap_text_embeddings_normalized.to(device) } )

# todo make riffusion generation and vocoder


decoder_config_path = 'configs/train_decoder_config.audio_inference.json'

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

audio_embedds_normalized = audio_embedds / audio_embedds.norm(p=2, dim=-1, keepdim=True)
examples = [
    [ torch.rand([ 1, 80, 512 ]).to(device), audio_embedds_normalized[0, :].to(device), None, "" ],
    # [ torch.rand([ 1, 80, 512 ]), audio_embedds[0, :], None, "" ],
]

real_images, generated_images, captions = generate_samples(trainer, examples, device=device, match_image_size=False)

print(generated_images[0].shape)

generated_image = generated_images[0].detach().cpu().numpy()
real_image =real_images[0].detach().cpu().numpy()

gen_melspec = np.array(spectral_normalize_torch(torch.from_numpy(generated_image)))
target_melspec = np.array(spectral_normalize_torch(torch.from_numpy(real_image)))

np.save(".decoder/melspec_gen"+ ( "_prior" if do_clap_evaluation else "_pregen" ) + "_" + input_text.replace(" ", "_").lower() + ".npy", gen_melspec)

import matplotlib.pyplot as plt

print("gen_melspec", gen_melspec.shape)

plt.title("gen melspec: " + captions[0])
plt.imshow(gen_melspec[0, :, :])
plt.savefig(".decoder/melspec_gen" + ( "_prior" if do_clap_evaluation else "_pregen" ) + "_" + input_text.replace(" ", "_").lower() + ".png")
plt.clf()


