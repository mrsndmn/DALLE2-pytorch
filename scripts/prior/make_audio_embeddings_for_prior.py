import json
import pandas as pd
import sys
import os
from tqdm.auto import tqdm

from multiprocessing import Pool
import time
import os

import argparse
import numpy as np
import torchaudio
import torch

import sys

sys.path.insert(0, "/home/dtarasov/workspace/hse-audio-dalle2/DALLE2-pytorch")
sys.path.insert(0, '/home/dtarasov/workspace/hse-audio-dalle2/transformers/src')


from transformers import ClapModel, AutoProcessor, ClapProcessor

# riffusion repo must be script working directory

parser = argparse.ArgumentParser(description="Dataset preparation script.")

args = parser.parse_args()

name = "laion/clap-htsat-unfused"

device = 'cuda' if torch.cuda.is_available() else 'cpu'

clap = ClapModel.from_pretrained(name).to(device).float()
# processor =  ClapProcessor.from_pretrained(name)
processor =  AutoProcessor.from_pretrained(name)

dataset = pd.read_csv("../audiocaps/dataset/train.csv")

audio_path = "../data/audiocaps_train/"
embeddings_dir = '../data/audiocaps_prior_train_embeddings/'
count_spectrograms = 0

if count_spectrograms > 0:
    dataset = dataset.head( count_spectrograms )


if not os.path.isdir(embeddings_dir):
    os.mkdir(embeddings_dir)

if not os.path.isdir(embeddings_dir + 'audio'):
    os.mkdir(embeddings_dir + 'audio')

if not os.path.isdir(embeddings_dir + 'text'):
    os.mkdir(embeddings_dir + 'text')


with torch.no_grad():
    full_metadata_file_path = embeddings_dir + "metadata.jsonl"

    with open(full_metadata_file_path, 'w') as f:
        for _, row in tqdm(dataset.iterrows(), total=len(dataset), desc='prepare metadata'):

            file_name = row['youtube_id'] + '.wav'
            audio_embedding_file_name = 'audio/' + row['youtube_id'] + '_audio.npy'
            text_embedding_file_name = 'text/' + row['youtube_id'] + '_text.npy'
            full_path_audio = audio_path + file_name
            audio_full_path_embedding = embeddings_dir + audio_embedding_file_name
            text_full_path_embedding = embeddings_dir + text_embedding_file_name

            if not os.path.isfile(full_path_audio):
                continue

            waveform, sample_rate = torchaudio.load(full_path_audio)

            expected_sample_rate = 48000
            if sample_rate != expected_sample_rate:
                waveform = torchaudio.functional.resample(waveform, orig_freq=sample_rate, new_freq=expected_sample_rate)
                sample_rate = expected_sample_rate

            waveform = waveform[0, :]
            waveform = waveform.numpy()
            # print("waveform.shape", waveform.shape)

            processed_inputs = processor(text=[row['caption']], audios=waveform, sampling_rate=sample_rate, return_tensors="pt")
            # processed_inputs = processor(text=[row['caption']], audios=waveform, sampling_rate=sample_rate, return_tensors="pt", padding=True)
            # processed_inputs_text = processor( return_tensors="pt", padding=True)
            # print("processed_inputs", processed_inputs)
            # print("processed_inputs", processed_inputs.input_features.shape)

            # clap_outputs = clap(**processed_inputs_text, **processed_inputs_audio)
            for k, v in processed_inputs.items():
                processed_inputs[k] = v.to(device)

            clap_outputs = clap(**processed_inputs)

            np.save(audio_full_path_embedding, clap_outputs.audio_embeds.detach().cpu().numpy())
            np.save(text_full_path_embedding, clap_outputs.text_embeds.detach().cpu().numpy())


            jsonline = {
                "audio_embedding_file_name": audio_embedding_file_name,
                "text_embedding_file_name": text_embedding_file_name,
                "caption": row['caption'],
            }

            f.write( json.dumps(jsonline) + "\n" )

    # todo maybe also save clap_outputs.text_model_output.last_hidden_state ?


