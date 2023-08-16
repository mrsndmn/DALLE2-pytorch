
import sys
sys.path.insert(0, '/home/dtarasov/workspace/hse-audio-dalle2/diffusers/src')

from generate_melspectrograms import get_melspectrogram_from_waveform
import numpy as np

import torchaudio
from diffusers import AudioLDMPipeline
from meldataset import spectral_normalize_torch
from train_decoder import create_dataloaders, TrainDecoderConfig


config = TrainDecoderConfig.from_json_path('configs/train_decoder_config.audio.full.json')
all_shards = list(range(config.data.start_shard, config.data.end_shard + 1))

dataloaders = create_dataloaders(
    available_shards=all_shards,
    audio_preproc = config.data.audio_preproc,
    train_prop = config.data.splits.train,
    val_prop = config.data.splits.val,
    test_prop = config.data.splits.test,
    n_sample_images=config.train.n_sample_images,

    **config.data.dict(),

    seed = config.seed,
)

dataloader = dataloaders['train']

batch = next(iter(dataloader))

melspectrogarm = batch['audio_melspec'][0]

print("melspectrogarm", melspectrogarm[:1, :10])

np.save("melspectrogarm_from_dataloader.npy", melspectrogarm.detach().cpu().numpy())

print("caption:", batch["txt"][0])

assert len(melspectrogarm.shape) == 3, f"melspectrogarm.shape={melspectrogarm.shape}"

melspectrogarm = melspectrogarm.permute(0, 2, 1)
melspectrogarm = spectral_normalize_torch(melspectrogarm)

assert melspectrogarm.shape == (1, 512, 64), f'vocoder shape is not ok {melspectrogarm.shape}'

audioLDMpipe = AudioLDMPipeline.from_pretrained( "cvssp/audioldm-s-full-v2", local_files_only=True )
restored_waveform = audioLDMpipe.vocoder( melspectrogarm ).detach().cpu()

out_filename = "train_dataloader_restored.wav"
torchaudio.save(out_filename, restored_waveform, 16000)

print("melspec saved", out_filename)
