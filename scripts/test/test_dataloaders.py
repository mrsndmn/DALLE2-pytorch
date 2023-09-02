
import pytest
from train_decoder import create_dataloaders, TrainDecoderConfig

def test_dataloader():

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

    for dataloader_name, dataloader in dataloaders.items():
        for i, batch in enumerate(dataloader):
            assert set(batch.keys()) == set(['audio_emb', 'audio_melspec', 'txt']), f'{dataloader_name}[{i}] keys are ok'
            assert len(batch['txt']) == batch['audio_melspec'].shape[0], f'{dataloader_name}[{i}] txt length is ok'
