
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

do_clap_evaluation = False

input_text = "Person is whistling"

name = "laion/clap-htsat-unfused"

device = 'cuda' if torch.cuda.is_available() else 'cpu'

decoder_config_path = 'configs/train_decoder_config.audio_inference.json'

from train_decoder import create_tracker, recall_trainer, generate_samples
from dalle2_pytorch.trainer import DecoderTrainer
from accelerate import Accelerator

from torchvision.transforms import Resize

accelerator = Accelerator()
config = TrainDecoderConfig.from_json_path(str(decoder_config_path))
tracker = create_tracker(accelerator, config, decoder_config_path, dummy=False)

decoder = config.decoder.create()
decoder.to(device)

trainer = DecoderTrainer(
    decoder=decoder,
    accelerator=accelerator,
)

if tracker.can_recall:
    recall_trainer(tracker, trainer)
else:
    print("\n\n\n!!!!NO RECALL WAS CALLED!!!!\n\n\n")

# audio_embedds_normalized = audio_embedds / audio_embedds.norm(p=2, dim=-1, keepdim=True)
# examples = [
#     [ torch.rand([ 1, 16, 128 ]), audio_embedds[0, :], None, "" ],
#     [ torch.rand([ 1, 16, 128 ]), audio_embedds_normalized[0, :], None, "" ],
# ]

config = TrainDecoderConfig.from_json_path(str(decoder_config_path))
all_shards = list(range(config.data.start_shard, config.data.end_shard + 1))
world_size = 1
rank = 0
shards_per_process = len(all_shards) // world_size
assert shards_per_process > 0, "Not enough shards to split evenly"
my_shards = all_shards[rank * shards_per_process: (rank + 1) * shards_per_process]

print("my_shards", my_shards)

dataloaders = create_dataloaders(
    available_shards=my_shards,
    audio_preproc = config.data.audio_preproc,
    train_prop = config.data.splits.train,
    val_prop = config.data.splits.val,
    test_prop = config.data.splits.test,
    n_sample_images=config.train.n_sample_images,
    # webdataset_base_url=config.data.webdataset_base_url, # should be in config
    **config.data.dict(),
    rank = rank,
    seed = config.seed,
)


n_samples = 10
examples = get_example_data(dataloaders["train_sampling"], device, n_samples)

print("examples", len(examples))

real_images, generated_images, captions = generate_samples(trainer, examples, device=device, match_image_size=False)

for i in range(len(generated_images)):

    # gen_melspec = np.array(spectral_normalize_torch(resized_generated_image))
    # target_melspec = np.array(spectral_normalize_torch(resized_real_image))

    gen_melspec = spectral_normalize_torch(generated_images[i]).detach().cpu().numpy()
    target_melspec = spectral_normalize_torch(real_images[i]).detach().cpu().numpy()

    caption_lower = captions[i].replace(" ", "_").lower()

    np.save(".decoder/melspec_gen_" + str(i) + "_" + caption_lower + ( "_prior" if do_clap_evaluation else "_pregen" ) +".npy", gen_melspec)
    np.save(".decoder/melspec_tgt_" + str(i) + "_" + caption_lower + ( "_prior" if do_clap_evaluation else "_pregen" ) +".npy", target_melspec)

    import matplotlib.pyplot as plt


    plt.title("gen melspec: " + captions[i])
    plt.imshow(gen_melspec[0, :, :])
    plt.savefig(".decoder/melspec_gen" + str(i) + ( "_prior" if do_clap_evaluation else "_pregen" ) + ".png")
    plt.clf()

    plt.title("tgt melspec: " + captions[i])
    plt.imshow(target_melspec[0, :, :])
    plt.savefig(".decoder/melspec_tgt" + str(i) + ( "_prior" if do_clap_evaluation else "_pregen" ) + ".png")
    plt.clf()

