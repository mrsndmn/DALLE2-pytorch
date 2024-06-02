
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

import torchaudio
import webdataset as wds

sys.path.insert(0, "/home/dtarasov/workspace/hse-audio-dalle2/DALLE2-pytorch")
sys.path.insert(0, "/home/dtarasov/workspace/hse-audio-dalle2/hifi-gan")
sys.path.insert(0, '/home/dtarasov/workspace/hse-audio-dalle2/transformers/src')

from transformers import spectrogram

from meldataset import MelDataset, mel_spectrogram, get_dataset_filelist


nfft = 1024
num_mels = 64
hop_size = 160
win_size = 1024
fmin = 0
fmax = 8000

sample_rate, waveform = torchaudio.load("filename.wav")

expected_melspec_samplerate = 16000
waveform_resampled = waveform[:1, :]
if sample_rate != expected_melspec_samplerate:
    waveform_resampled = torchaudio.functional.resample(waveform, orig_freq=sample_rate, new_freq=expected_melspec_samplerate)

hifigan_mel = mel_spectrogram(waveform_resampled, nfft, num_mels, sample_rate, hop_size, win_size, fmin, fmax, center=False)
hifigan_mel = torch.exp(hifigan_mel)

# 
t5feature_extractor = SpeechT5FeatureExtractor(
    
)

self._extract_mel_features(waveform_resampled)

transformers_mel = spectrogram(
    waveform_resampled,
    window=win_size,
    frame_length=self.sample_size,
    hop_length=hop_size,
    fft_length=nfft,
    mel_filters=num_mels,
    # mel_floor=self.mel_floor,
    log_mel="log10",
)