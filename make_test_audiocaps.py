

from audio_dalle_full_inference import audio_dalle2_full_inference

import os
import pandas as pd

csv_path = "../audiocaps/dataset/test.csv"
dataset = pd.read_csv(csv_path)

data_dir = '../data/audiocaps_test'

audio_dir_files = set(os.listdir(data_dir))

batch = []

MAX_BATCH_SIZE = 50

for x in dataset.iterrows():
    file_name = x['youtube_id'] + ".wav"

    if file_name in audio_dir_files:
        batch.append({
            "id":      x['youtube_id'],
            "caption": x['caption'],
        })

        if len(batch) >= MAX_BATCH_SIZE:
            audio_dalle2_full_inference(batch)
            batch = []

if len(batch) > 0:
    audio_dalle2_full_inference(batch, audio_dalle2_full_inference='../audioldm_eval/audiodalle_test')
    batch = []

