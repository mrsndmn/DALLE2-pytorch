

from audio_dalle_full_inference import audio_dalle2_full_inference

import os
import pandas as pd
import argparse

parser = argparse.ArgumentParser()

parser.add_argument( "--config-file", type=str, required=True, help="Path to decoder conifg" )

parser.add_argument( "--limit-samples", type=str, required=False, help="Generated samples limit", default=1000 )

args = parser.parse_args()

decoder_config_path = args.config_file
limit_samples = args.limit_samples

csv_path = "../audiocaps/dataset/test.csv"
dataset = pd.read_csv(csv_path)

data_dir = '../data/audiocaps_test'

audio_dir_files = set(os.listdir(data_dir))

batch = []

MAX_BATCH_SIZE = 100

for _, x in dataset.iterrows():
    file_name = x['youtube_id'] + ".wav"

    if limit_samples <= 0:
        break

    if file_name in audio_dir_files:
        batch.append({
            "id":      x['youtube_id'],
            "caption": x['caption'],
        })

        limit_samples -= 1

        if len(batch) >= MAX_BATCH_SIZE:
            audio_dalle2_full_inference(batch, decoder_config_path=decoder_config_path)
            batch = []

if len(batch) > 0:
    audio_dalle2_full_inference(batch, decoder_config_path=decoder_config_path)

