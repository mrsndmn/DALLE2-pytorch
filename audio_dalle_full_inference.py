
import sys
import os
import torch

sys.path.insert(0, '/home/dtarasov/workspace/hse-audio-dalle2/diffusers/src')
sys.path.insert(0, '/home/dtarasov/workspace/hse-audio-dalle2/transformers/src')
sys.path.insert(0, '/home/dtarasov/workspace/hse-audio-dalle2/DALLE2-pytorch')
sys.path.insert(0, '/home/dtarasov/workspace/hse-audio-dalle2/hifi-gan')

from diffusers import AudioLDMPipeline

from dalle2_pytorch.train_configs import TrainDecoderConfig

from transformers import ClapTextModelWithProjection, AutoTokenizer

from dalle2_pytorch import DiffusionPrior
from dalle2_pytorch.train_configs import TrainDiffusionPriorConfig

from meldataset import spectral_normalize_torch

from inference_utils import save_melspec

from train_decoder import create_tracker, recall_trainer, generate_samples
from dalle2_pytorch.trainer import DecoderTrainer
from accelerate import Accelerator


def audio_dalle2_full_inference(input_data, generated_output_dir=None):

    input_texts = [ x["caption"] for x in input_data ]

    name = "laion/clap-htsat-unfused"

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    audioLDMpipe = AudioLDMPipeline.from_pretrained( "cvssp/audioldm-s-full-v2", local_files_only=True )
    audioLDMpipe.to(device)

    print("device", device)

    print("getting model")
    clap = ClapTextModelWithProjection.from_pretrained(name).float()
    # processor =  ClapProcessor.from_pretrained(name)
    print("getting tokenizer")
    tokenizer =  AutoTokenizer.from_pretrained(name)

    print("run tokenizer prompt", input_texts)
    processed_inputs_text = tokenizer(text=input_texts, padding=True, return_tensors="pt")

    print("run clap")
    clap_text_outputs = clap(**processed_inputs_text)

    print("clap_text_outputs.text_embeds.shape", clap_text_outputs.text_embeds.shape) # [ 1, 512 ]

    prior_train_state = torch.load('.prior/latest_checkpoint.pth', map_location=device)

    config = TrainDiffusionPriorConfig.from_json_path('configs/train_clap_prior_config.json')

    diffusionPrior: DiffusionPrior = config.prior.create()
    diffusionPrior.load_state_dict(prior_train_state['model'])
    diffusionPrior.to(device)

    clap_text_embeddings_normalized = clap_text_outputs.text_embeds / clap_text_outputs.text_embeds.norm(p=2, dim=-1, keepdim=True)

    audio_embedds = diffusionPrior.p_sample_loop( clap_text_outputs.text_embeds.shape, text_cond = { "text_embed": clap_text_embeddings_normalized.to(device) } )

    decoder_base_path = '.decoder_full_inference'

    if not os.path.exists(decoder_base_path):
        os.mkdir(decoder_base_path)

    if not os.path.exists(decoder_base_path + '/full_inference'):
        os.mkdir(decoder_base_path + '/full_inference')

    decoder_config_path = 'configs/train_decoder_config.audio.full_inference.json'

    accelerator = Accelerator()
    config = TrainDecoderConfig.from_json_path(str(decoder_config_path))
    tracker = create_tracker(accelerator, config, decoder_config_path, dummy=False)

    decoder = config.decoder.create()

    trainer = DecoderTrainer(
        decoder=decoder,
        accelerator=accelerator,
    )
    trainer.to(device)

    if tracker.can_recall:
        recall_trainer(tracker, trainer)
    else:
        raise Exception("\n\n\n!!!!NO RECALL WAS CALLED!!!!\n\n\n")

    audio_embedds_normalized = audio_embedds / audio_embedds.norm(p=2, dim=-1, keepdim=True)

    examples = []
    for i in range(audio_embedds_normalized.shape[0]):
        examples.append([ torch.rand([ 1, 64, 512 ]).to(device), audio_embedds_normalized[i, :].to(device), None, input_texts[i] ],)

    real_images, generated_images, captions = generate_samples(trainer, examples, device=device, match_image_size=False)

    if generated_output_dir is None:
        generated_output_dir = decoder_base_path + "/decoder_inference"

    for i, input_text in enumerate(input_texts):
        file_id = input_data[i]['id']

        generated_image = generated_images[i]
        real_image =real_images[i]

        gen_melspec_t = spectral_normalize_torch(generated_image).detach()

        save_melspec(audioLDMpipe, generated_output_dir, gen_melspec_t, input_text, melspec_type="gen", file_prefix=file_id)

    print("done")


if __name__ == '__main__':

    input_data = [
        {"id": "0", "caption": "A duck quacks"},
        {"id": "1", "caption": "A cat is meowing"},
        {"id": "2", "caption": "A cat meows"},
        {"id": "3", "caption": "Dogs bark nearby"},
        {"id": "4", "caption": "Birds chirping and tweeting"},
        {"id": "5", "caption": "A fly buzzes"},
        {"id": "6", "caption": "A man talks"},
        {"id": "7", "caption": "A man talks loudly"},
        {"id": "8", "caption": "A man talks silently"},
    ]

    audio_dalle2_full_inference(input_data)
