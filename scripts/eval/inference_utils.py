
import matplotlib.pyplot as plt
import torchaudio
import numpy as np

def save_melspec(audioLDMpipe, base_path_prefix, melspec_t, input_text, file_prefix=None, melspec_type="gen"):

    if file_prefix is None:
        file_prefix = input_text.replace(" ", "_").lower()

    mepspec = melspec_t.cpu().numpy()

    np.save(base_path_prefix + "/melspec_" + melspec_type + "_" + file_prefix + ".npy", mepspec)

    plt.title(melspec_type + " melspec: " + input_text)
    plt.imshow(mepspec[0, :, :])
    plt.savefig(base_path_prefix + "/melspec_" + melspec_type  + "_" + file_prefix + ".png")
    plt.clf()

    generated_image_for_vocoder = melspec_t.permute(0, 2, 1)
    assert generated_image_for_vocoder.shape == (1, 512, 64), f'vocoder shape is not ok {generated_image_for_vocoder.shape}'

    sample_waveform = audioLDMpipe.vocoder(generated_image_for_vocoder).detach().cpu()

    result_wav_file = base_path_prefix + "/" + melspec_type + file_prefix + ".wav"
    torchaudio.save( result_wav_file, sample_waveform, 16000 )
    print("saved wav:", result_wav_file)

    return

