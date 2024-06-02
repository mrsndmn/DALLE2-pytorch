
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
import argparse

from dalle2_pytorch.inference import audio_dalle_decoder_generate

import json

def make_inference_config(decoder_config_path: str):
    with open(decoder_config_path, 'r') as f:
        decoder_config = json.load(f)

    if 'inference' in decoder_config_path:
        raise Exception("audio_dalle_full_inference.py accepts training config and build inference config by itself. Your config file must not contain `inference` word")

    decoder_config['tracker']['log']['wandb_project'] = decoder_config['tracker']['log']['wandb_project'] + '_inference'

    inference_datapath = decoder_config['tracker']['data_path'] + '_inference'
    decoder_config['tracker']['data_path'] = inference_datapath
    if not os.path.isdir(inference_datapath):
        os.mkdir(inference_datapath)

    decoder_config['tracker']['load']['load_from'] = 'local'
    decoder_config['tracker']['load']['file_path'] = decoder_config['tracker']['save'][0]['save_latest_to']

    decoder_config['tracker']['save'][0]['save_latest_to'] = os.path.join(inference_datapath, 'latest_checkpoint.pth')
    decoder_config['tracker']['save'][0]['save_best_to'] = os.path.join(inference_datapath, 'best_checkpoint.pth')

    print("updated tracker config:", decoder_config['tracker'])

    decoder_config_basename = os.path.basename(decoder_config_path)
    decoder_inference_config_path = os.path.join('configs', 'inference', decoder_config_basename)

    with open(decoder_inference_config_path, 'w') as f:
        json.dump(decoder_config, f, indent=4)

    return decoder_config, decoder_inference_config_path


def dynamic_range_compression_torch(x, C=1, clip_val=1e-5, normalize_fn=torch.log):
    return normalize_fn(torch.clamp(x, min=clip_val) * C)


def audio_dalle2_full_inference(
        input_data, # [ { id: "xxx", text: "blabla" } ]
        decoder_configs_path=None,
        inference_out_base_path=None
    ):

    if decoder_configs_path is None:
        raise Exception("decoder_configs_path can't be None")

    if inference_out_base_path is None:
        raise Exception("inference_out_base_path can't be None")

    # decoder_config, decoder_inference_config_path = make_inference_config(decoder_configs_path)
    # decoder_base_path = decoder_config['tracker']['data_path']

    input_texts = [ x["caption"] for x in input_data ]
    input_youtube_ids = [ x["id"] for x in input_data ]

    name = "laion/clap-htsat-unfused"

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    audioLDMpipe = AudioLDMPipeline.from_pretrained( "cvssp/audioldm-s-full-v2", local_files_only=True )
    audioLDMpipe.to(device)

    print("device", device)

    print("getting model")
    clap = ClapModel.from_pretrained(name).to(device).float()
    # processor =  ClapProcessor.from_pretrained(name)
    print("getting tokenizer")
    tokenizer =  AutoTokenizer.from_pretrained(name)

    print("run tokenizer prompt", input_texts)
    processed_inputs_text = tokenizer(text=input_texts, padding=True, return_tensors="pt")
    for k, v in processed_inputs_text.items():
        processed_inputs_text[k] = v.to(device)

    clap_text_embeddings_normalized = clap.get_text_features(input_ids=processed_inputs_text['input_ids'], attention_mask=processed_inputs_text['attention_mask'])

    audio_embedds = clap_text_embeddings_normalized

    if not os.path.exists(inference_out_base_path):
        os.mkdir(inference_out_base_path)

    audio_embedds_normalized = audio_embedds / audio_embedds.norm(p=2, dim=-1, keepdim=True)

    # audio_embedds_normalized = audio_embedds

    examples = []
    for i in range(audio_embedds_normalized.shape[0]):
        examples.append([ torch.rand([ 1, 64, 512 ]).to(device), audio_embedds_normalized[i, :].to(device), None, input_texts[i], input_youtube_ids[i] ],)

    audio_dalle_decoder_generate(audioLDMpipe, decoder_configs_path, examples, inference_out_base_path, device=device)

    print("done")

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument( "--config_file", type=str, required=True, help="Path to decoder conifg" )
    parser.add_argument( "--output_dir", type=str, required=True, help="Path to decoder conifg" )

    args = parser.parse_args()

    config = args.config_file
    output_dir = args.output_dir

    input_data = [
        {"id": "0", "caption": "A duck quacks multiple times"},
        {"id": "1", "caption": "A cat is meowing"},
        {"id": "2", "caption": "A cat meows"},
        {"id": "3", "caption": "Dogs bark nearby"},
        {"id": "4", "caption": "Birds chirping and tweeting"},
        {"id": "5", "caption": "A fly buzzes"},
        {"id": "6", "caption": "A man talks"},
        {"id": "7", "caption": "A man talks loudly"},
        {"id": "8", "caption": "A man talks silently"},
    ]

    audio_dalle2_full_inference(
        input_data,
        decoder_config_path=config,
        inference_out_base_path=output_dir
    )
