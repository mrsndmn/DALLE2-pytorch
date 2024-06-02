import sys

sys.path.insert(0, '/home/dtarasov/workspace/hse-audio-dalle2/AudioLDM')


import torch
from audioldm import text_to_audio, build_model
import torchaudio
from tqdm.auto import tqdm

import os
import pandas as pd


csv_path = "../audiocaps/dataset/test.csv"
dataset = pd.read_csv(csv_path)

data_dir = '../data/audiocaps_test'
target_path = '.audioldm_test_generated/'

audio_dir_files = set(os.listdir(data_dir))
target_existing_files = set(os.listdir(target_path))

model_name = 'audioldm-s-full'

audioldm=build_model(model_name=model_name)
current_model_name = model_name

random_seed = 42

for _, x in tqdm(dataset.iterrows()):
    file_name = x['youtube_id'] + ".wav"

    if file_name not in audio_dir_files:
        continue

    if file_name in target_existing_files:
        continue

    print("do ", file_name)
    waveform = text_to_audio(
        latent_diffusion=audioldm,
        text=x['caption'],
        seed=random_seed,
        duration=5,
        guidance_scale=3,
        n_candidate_gen_per_text=int(1),
    )  # [bs, 1, samples]

    torchaudio.save(target_path + file_name, torch.tensor(waveform[0]), 16000)


