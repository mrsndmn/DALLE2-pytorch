
import sys
sys.path.insert(0, '/home/dtarasov/workspace/hse-audio-dalle2/diffusers/src')

from scripts.data.decoder_prepare_dataset import get_melspectrogram_from_waveform
import numpy as np

from scripts.eval.inference_utils import save_melspec

import torchaudio
from diffusers import AudioLDMPipeline
from meldataset import spectral_normalize_torch
from train_decoder import create_dataloaders, TrainDecoderConfig

from train_decoder import create_dataloaders, get_example_data


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

device = 'cpu'
examples = get_example_data(dataloader, device, 10)

i = 0
for melspectrogarm, img_embeddings, text_embeddings, input_text in examples:

    # np.save(f"melspectrogarm_from_dataloader{i}.npy", melspectrogarm.detach().cpu().numpy())

    assert len(melspectrogarm.shape) == 3, f"melspectrogarm.shape={melspectrogarm.shape}"

    audioLDMpipe = AudioLDMPipeline.from_pretrained( "cvssp/audioldm-s-full-v2", local_files_only=True )

    target_melspec_t = spectral_normalize_torch( melspectrogarm ).detach()

    decoder_base_path = '.test_voceder'
    save_melspec(audioLDMpipe, decoder_base_path, target_melspec_t, input_text, melspec_type=f"test_vocoder_{i}__")

    i += 1

