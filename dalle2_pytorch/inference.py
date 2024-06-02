from train_decoder import create_tracker, recall_trainer, generate_samples
from meldataset import spectral_normalize_torch
from scripts.eval.inference_utils import save_melspec
from accelerate import Accelerator
from dalle2_pytorch.train_configs import TrainDecoderConfig
from dalle2_pytorch.trainer import DecoderTrainer


def audio_dalle_decoder_generate(audioLDMpipe, configs, examples, inference_out_base_path:str, device='cuda'):

    print("configs", configs)

    clip = None
    start_unet = 1
    end_unet = len(configs)
    condition_on_text_encodings = False # todo is it?
    cond_scale = 1.0
    text_prepend = ""

    generated_images = None

    for unet_idx in range(start_unet, end_unet + 1):
        print("unet_idx", unet_idx)

        accelerator = Accelerator()
        decoder_config_path = configs[unet_idx - 1] # 'configs/inference/train_decoder_config.audio.full_no_prior_A100_u1.json'

        config = TrainDecoderConfig.from_json_path(str(decoder_config_path))

        decoder = config.decoder.create()

        trainer = DecoderTrainer(
            decoder=decoder,
            accelerator=accelerator,
        )
        trainer.to(device)
        trainer.eval()

        tracker = create_tracker(accelerator, config, decoder_config_path, dummy=False)

        if tracker.can_recall:
            recall_trainer(tracker, trainer, strict=False)
        else:
            raise Exception("\n\n\n!!!!NO RECALL WAS CALLED!!!!\n\n\n")


        if unet_idx > 1:
            assert generated_images is not None, f"generated_images is none: {unet_idx}"

            real_images, img_embeddings, text_embeddings, txts, youtube_ids = zip(*examples)
            # hack generated images
            examples = list(zip(generated_images, img_embeddings, text_embeddings, txts, youtube_ids))


        real_images, generated_images, captions, youtube_ids = generate_samples(trainer, examples, clip, unet_idx, unet_idx, condition_on_text_encodings, cond_scale, device, text_prepend, match_image_size=True)

        for real_image, generated_image, input_text, youtube_id in zip(real_images, generated_images, captions, youtube_ids):

            print(f"{unet_idx} generated_image.shape ", generated_image.shape)
            print(f"{unet_idx} real_image.shape      ", real_image.shape)

            gen_melspec_t = spectral_normalize_torch(generated_image).detach()
            target_melspec_t = spectral_normalize_torch(real_image).detach()

            save_melspec(audioLDMpipe, inference_out_base_path, gen_melspec_t, input_text, file_prefix=youtube_id, melspec_type=f"{unet_idx}gen_")
            save_melspec(audioLDMpipe, inference_out_base_path, target_melspec_t, input_text, file_prefix=youtube_id, melspec_type=f"{unet_idx}tgt_")


