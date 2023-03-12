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

from transformers import ClapModel, AutoProcessor, ClapProcessor

# riffusion repo must be script working directory

parser = argparse.ArgumentParser(description="Dataset preparation script.")

args = parser.parse_args()

name = "laion/clap-htsat-fused"

device = 'cuda' if torch.cuda.is_available() else 'cpu'

clap = ClapModel.from_pretrained(name).to(device).float()
# processor =  ClapProcessor.from_pretrained(name)
processor =  AutoProcessor.from_pretrained(name)

dataset = pd.read_csv("../audiocaps/dataset/train.csv")

audio_dir = "../audiocaps_train/"
embeddings_dir = '../audiocaps_train_embeddings_1k/'
count_spectrograms = 1000

if count_spectrograms > 0:
    dataset = dataset.head( count_spectrograms )


if not os.path.isdir(embeddings_dir):
    os.mkdir(embeddings_dir)

# prepare spectrogram


# to prepare spectrogram dataset with cli
# ~/anaconda3_new/envs/riffusion/bin/python3.9 -m riffusion.cli audio-to-image --audio ../audiocaps_train/000AjsqXq54.wav --image ./000AjsqXq54.jpg

with torch.no_grad():
    full_metadata_file_path = embeddings_dir + "metadata.jsonl"
    with open(full_metadata_file_path, 'w') as f:
        for _, row in tqdm(dataset.iterrows(), total=len(dataset), desc='prepare metadata'):

            file_name = row['youtube_id'] + '.wav'
            audio_embedding_file_name = row['youtube_id'] + '_audio.npy'
            text_embedding_file_name = row['youtube_id'] + '_text.npy'
            full_path_audio = "../audiocaps_train/" + file_name
            audio_full_path_embedding = embeddings_dir + audio_embedding_file_name
            text_full_path_embedding = embeddings_dir + text_embedding_file_name

            if not os.path.isfile(full_path_audio):
                continue

            waveform, sample_rate = torchaudio.load(full_path_audio)

            expected_sample_rate = 48000
            if sample_rate != expected_sample_rate:
                waveform = torchaudio.functional.resample(waveform, orig_freq=sample_rate, new_freq=expected_sample_rate)
                sample_rate = expected_sample_rate

            processed_inputs = processor(text=row['caption'], audios=waveform, sampling_rate=sample_rate, return_tensors="pt", padding=True)
            # processed_inputs_text = processor( return_tensors="pt", padding=True)

            # clap_outputs = clap(**processed_inputs_text, **processed_inputs_audio)
            clap_outputs = clap(**processed_inputs)

            np.save(audio_embedding_file_name, clap_outputs.audio_embeds.detach().cpu().numpy())
            np.save(text_embedding_file_name, clap_outputs.text_embeds.detach().cpu().numpy())

            # todo maybe also save clap_outputs.text_model_output.last_hidden_state ?

            jsonline = {
                "audio_embedding_file_name": audio_embedding_file_name,
                "text_embedding_file_name": text_embedding_file_name,
                "caption": row['caption'],
            }

            f.write( json.dumps(jsonline) + "\n" )

    print("metadata file is prepared", full_metadata_file_path)




# # pooler_output = text_outputs.pooler_output
# # last_hidden_state = text_outputs.last_hidden_state
# # last_hidden_state = last_hidden_state[1 != processed_text["attention_mask"]] = 0

# #         return EmbeddedText(l2norm(text_embed), text_encodings)


# # todo убрать хардкод sample rate
# processed_audio = processor(, return_tensors="pt", padding=True)
# audio_outputs = clap.audio_model(**processed_audio)

#         return EmbeddedImage(l2norm(image_embed), image_encodings)
