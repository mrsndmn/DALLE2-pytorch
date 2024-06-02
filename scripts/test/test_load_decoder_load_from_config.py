import pytest
from dalle2_pytorch.train_configs import DecoderConfig, TrainDecoderConfig

from dalle2_pytorch.trackers import LocalLoader

def test_local_loader_is_ok():

    config = TrainDecoderConfig.from_json_path('configs/train_decoder_config.audio.u1.json')

    tracker = config.tracker.create(config, {})

    print("loder", tracker.loader)
    assert isinstance(tracker.loader, LocalLoader), "local loader loaded"
    assert str(tracker.loader.file_path) == '.decoder_u0_18.07/latest_checkpoint.pth', "local loader file path is ok"
