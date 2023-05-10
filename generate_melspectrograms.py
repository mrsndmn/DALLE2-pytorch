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


sys.path.append("/home/dtarasov/workspace/hse-audio-dalle2/hifi-gan")

from meldataset import MelDataset, mel_spectrogram, get_dataset_filelist

import numpy as np
from tqdm.auto import tqdm

# riffusion repo must be script working directory

parser = argparse.ArgumentParser(description="Dataset preparation script.")
parser.add_argument("--prepare-metadata", default=False, action='store_true')
parser.add_argument("--prepare-spectrograms", default=False, action='store_true')

args = parser.parse_args()

dataset = pd.read_csv("../audiocaps/dataset/train.csv")

audio_dir = "../data/audiocaps_train/"
audio_melspectrogram_dir = "../data/audiocaps_train_embeddings_1k/melspectrograms/"
audio_embeddings_dir = "../data/audiocaps_train_embeddings_1k/audio_embeddings/"

orig_audio_embeddings_dir = "../data/audiocaps_train_embeddings_1k/audio/"

webdataset_dir = "../data/audiocaps_train_embeddings_1k/webdataset/"

webdataset_tar = "../data/audiocaps_train_embeddings_1k/webdataset-0000.tar"

couny_spectrograms = 1000
# couny_spectrograms = 0

if couny_spectrograms > 0:
    dataset = dataset.head( couny_spectrograms )

if not os.path.isdir(audio_melspectrogram_dir):
    os.mkdir(audio_melspectrogram_dir)

if not os.path.isdir(webdataset_dir):
    os.mkdir(webdataset_dir)

if not os.path.isdir(audio_embeddings_dir):
    os.mkdir(audio_embeddings_dir)

# prepare spectrogram

audio_dir_files = set(os.listdir(audio_dir))
orig_audio_embeddings_files = set(os.listdir(orig_audio_embeddings_dir))
audio_embeddings_files = set(os.listdir(audio_embeddings_dir))
mel_spectrogram_files = set(os.listdir(audio_melspectrogram_dir))

nfft = 1024
num_mels = 80
hop_size = 256
win_size = 1024
fmin = 0
fmax = 8000

shard_num = '0000'

sink = wds.TarWriter(webdataset_tar)

def process_dataset_idx(i):
    try:
        row = dataset.iloc[i]

        file_name = row['youtube_id'] + '.wav'
        orig_audio_embedding_file_name = row['youtube_id'] + "_audio.npy"

        audio_embedding_file_name = shard_num + "{:04d}".format(i) + '.npy'
        melspectrogram_file_name = shard_num + "{:04d}".format(i) + '.npy'

        webdataset_audio_embedding_file_name = shard_num + "{:04d}".format(i) + '.audo_embedding.npy'
        webdataset_melspectrogram_file_name  = shard_num + "{:04d}".format(i) + '.melspectrogram.npy'

        if orig_audio_embedding_file_name not in orig_audio_embeddings_files:
            print("orig_audio_embedding_file_name not foud", orig_audio_embedding_file_name)
            return

        if file_name not in audio_dir_files:
            print("file_name not foud", file_name)
            return

        orig_path_for_audio_embeggins = orig_audio_embeddings_dir + orig_audio_embedding_file_name
        target_path_for_audio_embeggins = audio_melspectrogram_dir + melspectrogram_file_name
        target_webdataset_path_for_audio_embeggins = webdataset_dir + webdataset_audio_embedding_file_name
        # shutil.copyfile(orig_path_for_audio_embeggins, target_path_for_audio_embeggins)
        # shutil.copyfile(orig_path_for_audio_embeggins, target_webdataset_path_for_audio_embeggins)

        fill_path_audio = audio_dir + file_name

        waveform, sample_rate = torchaudio.load(fill_path_audio)

        waveform = waveform[:1, :]

        mel = mel_spectrogram(waveform, nfft, num_mels, sample_rate, hop_size, win_size, fmin, fmax, center=False)
        mel = torch.exp(mel)

        full_audio_embedding_file_name = audio_embeddings_dir + audio_embedding_file_name
        # np.save(full_audio_embedding_file_name, np.array(mel))

        full_webdataset_melspectrogram_file_name = webdataset_dir + webdataset_melspectrogram_file_name
        # np.save(full_webdataset_melspectrogram_file_name, np.array(mel))


        sink.write({
            "__key__": "sample{:04d}".format(i),
            "melspectrogram.npy": np.array(mel),
            "audio_emb.npy": np.load(orig_path_for_audio_embeggins),
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

print("sprctrogram files are prepared:", audio_melspectrogram_dir)
print("audio embedding files are prepared:", audio_embeddings_dir)

