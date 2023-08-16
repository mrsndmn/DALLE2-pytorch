
import sys
sys.path.insert(0, '/home/dtarasov/workspace/hse-audio-dalle2/diffusers/src')

from generate_melspectrograms import get_melspectrogram_from_waveform

import torchaudio
from diffusers import AudioLDMPipeline
from meldataset import spectral_normalize_torch

wav_filename = "bbMurLScYWA.wav"

# b28e4244b009bd533e20b4110688bc34  bbMurLScYWA.wav

waveform, samplerate = torchaudio.load(wav_filename)

melspectrogarm = get_melspectrogram_from_waveform(waveform, samplerate)
print("melspectrogarm.shape", melspectrogarm.shape)

melspectrogarm = melspectrogarm[:1, :, :]
melspectrogarm = spectral_normalize_torch(melspectrogarm)
melspectrogarm = melspectrogarm.permute(0, 2, 1)

print("melspectrogarm.shape", melspectrogarm.shape)

# assert melspectrogarm.shape == (1, 512, 64), f'vocoder shape is not ok {melspectrogarm.shape}'

audioLDMpipe = AudioLDMPipeline.from_pretrained( "cvssp/audioldm-s-full-v2", local_files_only=True )
restored_waveform = audioLDMpipe.vocoder( melspectrogarm ).detach().cpu()

out_filename = wav_filename + "melspec_restored.wav"
torchaudio.save(out_filename, restored_waveform, 16000)

print("wav saved", out_filename)