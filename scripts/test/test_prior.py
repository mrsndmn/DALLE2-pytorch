
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

device = 'cuda' if torch.cuda.is_available() else 'cpu'

name = "laion/clap-htsat-unfused"

tokenizer =  AutoTokenizer.from_pretrained(name)

prior_train_state = torch.load('.prior/latest_checkpoint.pth', map_location=device)

config = TrainDiffusionPriorConfig.from_json_path('configs/train_clap_prior_config.json')

diffusionPrior: DiffusionPrior = config.prior.create()
diffusionPrior.load_state_dict(prior_train_state['model'])
diffusionPrior.to(device)

clap = ClapModel.from_pretrained(name).to(device).float()

# data
prefix = '../data/audiocaps_prior_train_embeddings/'
audio_embedding_file_name = prefix + "audio/7v0S0E5ROXU_audio.npy",
text_embedding_file_name = prefix + "text/7v0S0E5ROXU_text.npy",
caption = "A duck quacks followed by small groans and beeps of small birds"

processed_inputs_text = tokenizer(text=[caption], padding=True, return_tensors="pt")
for k, v in processed_inputs_text.items():
    processed_inputs_text[k] = v.to(device)

clap_text_embeddings_normalized = clap.get_text_features(input_ids=processed_inputs_text['input_ids'], attention_mask=processed_inputs_text['attention_mask'])

audio_embedds = diffusionPrior.p_sample_loop( clap_text_embeddings_normalized.shape, text_cond = { "text_embed": clap_text_embeddings_normalized.to(device) } )

np.save('test_prior_duck_quacks_audio_sampled.npy', audio_embedds.detach().cpu().numpy())
np.save('test_prior_duck_quacks_test_clapped.npy', clap_text_embeddings_normalized.detach().cpu().numpy())



