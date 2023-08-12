
import sys
sys.path.insert(0, '/home/dtarasov/workspace/hse-audio-dalle2/diffusers/src')

from generate_melspectrograms import get_melspectrogram_from_waveform

import torchaudio
from diffusers import AudioLDMPipeline
from meldataset import spectral_normalize_torch

wav_filename = sys.argv[1]

waveform, samplerate = torchaudio.load(wav_filename)

melspectrogarm = get_melspectrogram_from_waveform(waveform, samplerate)
print("melspectrogarm.shape", melspectrogarm.shape)

melspectrogarm = spectral_normalize_torch(melspectrogarm)

melspectrogarm = melspectrogarm.permute(0, 2, 1)

audioLDMpipe = AudioLDMPipeline.from_pretrained( "cvssp/audioldm-s-full-v2", local_files_only=True )
restored_waveform = audioLDMpipe.vocoder( melspectrogarm ).detach().cpu()

torchaudio.save(wav_filename + "restored.wav", restored_waveform, 16000)

