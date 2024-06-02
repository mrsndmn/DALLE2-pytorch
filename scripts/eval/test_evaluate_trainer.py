import sys
import os

sys.path.insert(0, '/home/dtarasov/workspace/hse-audio-dalle2/diffusers/src')
sys.path.insert(0, '/home/dtarasov/workspace/hse-audio-dalle2/transformers/src')
sys.path.insert(0, '/home/dtarasov/workspace/hse-audio-dalle2/DALLE2-pytorch')
sys.path.insert(0, '/home/dtarasov/workspace/hse-audio-dalle2/hifi-gan')

from accelerate import Accelerator, DistributedDataParallelKwargs, InitProcessGroupKwargs

import torch
from train_decoder import evaluate_trainer, create_dataloaders, TrainDecoderConfig, DecoderTrainer

device = 'cuda' if torch.cuda.is_available() else 'cpu'


config = TrainDecoderConfig.from_json_path('configs/train_decoder_config.audio.full_no_prior_no_clap_lowrescond.json')

decoder = config.decoder.create()

decoder = decoder.to(device)

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

# for dataloader_name, dataloader in dataloaders.items():
#     print("dataloader_name", dataloader_name)
#     for i, batch in enumerate(dataloader):
#         if i >= 10:
#             break
#         print("barch i", i)
#         assert set(batch.keys()) == set(['audio_emb', 'audio_melspec', 'txt', 'youtube_id']), f'{dataloader_name}[{i}] keys are ok'
#         assert len(batch['txt']) == batch['audio_melspec'].shape[0], f'{dataloader_name}[{i}] txt length is ok'
#         assert len(batch['youtube_id']) == batch['audio_melspec'].shape[0], f'{dataloader_name}[{i}] txt length is ok'


accelerator = Accelerator()
trainer = DecoderTrainer(
    decoder=decoder,
    accelerator=accelerator,
    dataloaders=dataloaders,
    # **kwargs # todo?
)

assert config.evaluate.AUDIOLDM_EVAL is not None

evaluate_trainer(
    trainer,
    trainer.val_loader,
    device,
    0,
    len(config.train.unet_training_mask),
    clip=None,
    inference_device=device,
    random_generated_samples=True,
    **config.evaluate.dict(),
    condition_on_text_encodings=False,
    data_path=config.tracker.data_path
)

