import json
import pandas as pd
import sys
import os
from tqdm.auto import tqdm
import shutil
import torch

from multiprocessing import Pool
import time
import os

import argparse

import torchaudio
import webdataset as wds

sys.path.insert(0, "/home/dtarasov/workspace/hse-audio-dalle2/DALLE2-pytorch")
sys.path.insert(0, "/home/dtarasov/workspace/hse-audio-dalle2/hifi-gan")
sys.path.insert(0, '/home/dtarasov/workspace/hse-audio-dalle2/transformers/src')

from meldataset import MelDataset, mel_spectrogram, get_dataset_filelist

import numpy as np
from tqdm.auto import tqdm
from transformers import ClapModel, AutoProcessor, ClapProcessor

# riffusion repo must be script working directory

parser = argparse.ArgumentParser(description="Dataset preparation script.")
parser.add_argument("--prepare-metadata", default=False, action='store_true')
parser.add_argument("--prepare-spectrograms", default=False, action='store_true')

args = parser.parse_args()

name = "laion/clap-htsat-unfused"

device = 'cuda' if torch.cuda.is_available() else 'cpu'

clap = ClapModel.from_pretrained(name).to(device).float()
# processor =  ClapProcessor.from_pretrained(name)
processor =  AutoProcessor.from_pretrained(name)

dataset = pd.read_csv("../audiocaps/dataset/train.csv")

audio_dir = "../data/audiocaps_train/"

webdataset_dir = "../data/audiocaps_train_embeddings/webdataset/"

# couny_spectrograms = 1000
couny_spectrograms = 0

if couny_spectrograms > 0:
    dataset = dataset.head( couny_spectrograms )

if not os.path.isdir(webdataset_dir):
    os.mkdir(webdataset_dir)

audio_dir_files = set(os.listdir(audio_dir))

nfft = 1024
num_mels = 80
hop_size = 256
win_size = 1024
fmin = 0
fmax = 8000


webdataset_shard = 0
sink = None

def process_dataset_idx(i):

    current_shard = i // 1000
    global sink, webdataset_shard
    if sink is None or webdataset_shard != current_shard:
        webdataset_shard = current_shard
        webdataset_tar = webdataset_dir + "webdataset-{:03d}.tar".format(webdataset_shard)
        print("reopen dataset tar:", webdataset_tar)

        if sink is not None:
            sink.close()
        sink = wds.TarWriter(webdataset_tar)

    try:
        row = dataset.iloc[i]

        file_name = row['youtube_id'] + '.wav'

        if file_name not in audio_dir_files:
            print("file_name not foud", file_name)
            return

        fill_path_audio = audio_dir + file_name

        waveform, sample_rate = torchaudio.load(fill_path_audio)

        waveform = waveform[:1, :]
        expected_sample_rate = 48000 # хотя CLAP требует 4800 сэмпл рейт, этот декодер будет генерить 22050 sampling rate
        if sample_rate != expected_sample_rate:
            waveform = torchaudio.functional.resample(waveform, orig_freq=sample_rate, new_freq=expected_sample_rate)
            sample_rate = expected_sample_rate

        processed_inputs = processor(text=[row['caption']], audios=waveform[0, :].numpy(), sampling_rate=sample_rate, return_tensors="pt")
        # processed_inputs = processor(text=[row['caption']], audios=waveform, sampling_rate=sample_rate, return_tensors="pt", padding=True)
        # processed_inputs_text = processor( return_tensors="pt", padding=True)
        # print("processed_inputs", processed_inputs)
        # print("processed_inputs", processed_inputs.input_features.shape)

        # clap_outputs = clap(**processed_inputs_text, **processed_inputs_audio)
        for k, v in processed_inputs.items():
            processed_inputs[k] = v.to(device)

        clap_outputs = clap(**processed_inputs)

        expected_sample_rate = 16000 # хотя CLAP требует 4800 сэмпл рейт, этот декодер будет генерить 16000 sampling rate
        waveform22 = torchaudio.functional.resample(waveform, orig_freq=sample_rate, new_freq=expected_sample_rate)

        mel = mel_spectrogram(waveform22, nfft, num_mels, sample_rate, hop_size, win_size, fmin, fmax, center=False)
        mel = torch.exp(mel)

        sink.write({
            "__key__": "sample{:06d}".format(i),
            "melspectrogram.npy": np.array(mel),
            "audio_emb.npy": clap_outputs.audio_embeds.detach().cpu().numpy(),
            "text_emb.npy": clap_outputs.text_embeds.detach().cpu().numpy(),
            "txt": row['caption']
        })


    except Exception as e:
        print("cant process", i, "exception", e)

    return


print("prepare melspectrograms")

for i in tqdm(range(len(dataset))):
    process_dataset_idx(i)

sink.close()

# with Pool(processes=4) as pool:

#     result = list(tqdm(pool.imap(process_dataset_idx, range(len(dataset))), total=len(dataset), desc='prepare spectrograms'))

print("webdataset_dir files are prepared:", webdataset_dir)

