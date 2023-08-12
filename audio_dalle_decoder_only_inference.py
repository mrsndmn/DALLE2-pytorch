
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


def save_melspec(melspec_t, file_prefix, melspec_type="gen"):

    mepspec = melspec_t.cpu().numpy()

    base_path_prefix = decoder_base_path + "/decoder_inference"

    np.save(base_path_prefix + "/melspec_" + melspec_type + "_" + file_prefix + ".npy", mepspec)

    import matplotlib.pyplot as plt

    plt.title(melspec_type + " melspec: " + input_text)
    plt.imshow(mepspec[0, :, :])
    plt.savefig(base_path_prefix + "/melspec_" + melspec_type  + "_" + file_prefix + ".png")
    plt.clf()

    generated_image_for_vocoder = melspec_t.permute(0, 2, 1)
    assert generated_image_for_vocoder.shape == (1, 512, 64), f'vocoder shape is not ok {generated_image_for_vocoder.shape}'

    sample_waveform = audioLDMpipe.vocoder(generated_image_for_vocoder).detach().cpu()

    torchaudio.save( base_path_prefix + "/" + melspec_type + file_prefix + ".wav", sample_waveform, 16000 )

    return


for real_image, generated_image, input_text in zip(real_images, generated_images, captions):

    gen_melspec_t = spectral_normalize_torch(generated_image).detach()
    target_melspec_t = spectral_normalize_torch(real_image).detach()

    sample_file_suffix = input_text.replace(" ", "_").lower()

    save_melspec(gen_melspec_t, sample_file_suffix, melspec_type="gen")
    save_melspec(target_melspec_t, sample_file_suffix, melspec_type="tgt")


print("done")
