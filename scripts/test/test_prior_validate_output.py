
import sys
import os
import torch
import numpy as np

sys.path.insert(0, '/home/dtarasov/workspace/hse-audio-dalle2/diffusers/src')
sys.path.insert(0, '/home/dtarasov/workspace/hse-audio-dalle2/transformers/src')
sys.path.insert(0, '/home/dtarasov/workspace/hse-audio-dalle2/DALLE2-pytorch')
sys.path.insert(0, '/home/dtarasov/workspace/hse-audio-dalle2/hifi-gan')

from diffusers import AudioLDMPipeline

from dalle2_pytorch.train_configs import TrainDecoderConfig

from transformers import ClapModel, AutoTokenizer

from dalle2_pytorch import DiffusionPrior
from dalle2_pytorch.train_configs import TrainDiffusionPriorConfig

from meldataset import spectral_normalize_torch

from scripts.eval.inference_utils import save_melspec

from train_decoder import create_tracker, recall_trainer, generate_samples
from dalle2_pytorch.trainer import DecoderTrainer
from accelerate import Accelerator

import sys
sys.path.insert(0, '/home/dtarasov/workspace/hse-audio-dalle2/diffusers/src')

from scripts.data.decoder_prepare_dataset import get_melspectrogram_from_waveform

import torchaudio
from diffusers import AudioLDMPipeline
from meldataset import spectral_normalize_torch

clapped = np.load('test_prior_duck_quacks_test_clapped.npy')
sampled = np.load('test_prior_duck_quacks_audio_sampled.npy')

prefix = '../data/audiocaps_prior_train_embeddings/'
audio_embedding_file_name = prefix + "audio/7v0S0E5ROXU_audio.npy"
text_embedding_file_name = prefix + "text/7v0S0E5ROXU_text.npy"

dataset_audio_emb = np.load(audio_embedding_file_name)
dataset_text_emb = np.load(text_embedding_file_name)

print(clapped[:,:10])
print(dataset_text_emb[:,:10])

assert np.allclose(dataset_text_emb, clapped, atol=1e-3), 'text embeddings are ok'

print("l2 audio loss:", ((dataset_audio_emb - sampled)**2).sum())
