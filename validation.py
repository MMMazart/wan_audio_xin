import os
import torch
import logging
from PIL import Image
import gc

import wan
from utils.utils import save_videos_grid
from a2v_datasets.bucket_sampler import get_closest_ratio
from tools.audio_processor import AudioProcessor
from tools.face_detect import FaceDetector

whisper_path = "/mnt/data/public_ckpt/audio2feature/whisper-tiny"
det_path = "/mnt/data/public_ckpt/face_detect/scrfd_10g_shape640x640.onnx"

def a2v_validation(
    vae, text_encoder, clip_model, net,
    config, args, accelerator, global_step,
    testdata_list, aspect_ratios
):
    transformer3d_val = accelerator.unwrap_model(net).transformer3d
    audio2token_val = accelerator.unwrap_model(net).audio2token
    logging.info("transformer3d_val has been created.")
    logging.info("audio2token_val has been created.")
    
    pipeline = wan.WanA2V(
        config=config,
        checkpoint_dir=args.pretrained_model_name_or_path,
        device=accelerator.device,
        rank=None,
        text_encoder=text_encoder,
        vae=vae,
        model=transformer3d_val,
        audio2token=audio2token_val,
        validation_mode=True,
    )

    audio_processor = AudioProcessor(whisper_path=whisper_path)
    if not args.disable_mask:
        face_detector = FaceDetector(det_path=det_path, gpu_id=0)

    for i, (image_path, audio_path, text_prompt) in enumerate(testdata_list):

        with torch.no_grad():
            ref_image = Image.open(image_path).convert("RGB")
            img_height, img_width = int(ref_image.height), int(ref_image.width)
            closest_size, _ = get_closest_ratio(img_height, img_width, ratios=aspect_ratios)
            closest_size = [int(x / 8) * 8 for x in closest_size]

            new_height, new_width = closest_size
            ref_image = ref_image.resize((new_width, new_height), Image.Resampling.LANCZOS)

            if args.disable_mask:
                mask = None
            else:
                mask = face_detector.get_face_mask(ref_image)
                if mask is None:
                    print(f"{image_path} no face detected")
                    continue

            audio_emb = audio_processor.preprocess(audio_path)
            audio_prompts = audio_emb['audio_prompts']
            audio_len = audio_prompts.size(1) // 2

            audio_prompts = torch.cat([torch.zeros_like(audio_prompts[:,:4]), audio_prompts, torch.zeros_like(audio_prompts[:,:6])], 1)
            audio_prompts = audio_prompts.to(device=audio2token_val.device, dtype=audio2token_val.dtype)
            
            audio_list = []
            for k in range(audio_len):
                audio_list.append(audio_prompts[:, (k*2):(k*2+10)][0])
            
            if args.motion_frame_num == 0:
                sample = pipeline.generate(
                    input_prompt=text_prompt,
                    ref_images=[ref_image],
                    frame_num=args.video_sample_n_frames,
                    shift=5.0,
                    sample_solver='unipc',
                    sampling_steps=50,
                    offload_model=False,
                    audio_list=audio_list,
                    mask=mask,
                )
            else:
                sample = pipeline.generate_motion_frame(
                    input_prompt=text_prompt,
                    ref_images=[ref_image],
                    frame_num=args.video_sample_n_frames,
                    shift=5.0,
                    sample_solver='unipc',
                    sampling_steps=50,
                    offload_model=False,
                    audio_list=audio_list,
                    mask=mask,
                    motion_frame_num=args.motion_frame_num,
                )

            sample = sample.unsqueeze(0).cpu()

        save_path = os.path.join(args.output_dir, f"sample/sample-{global_step}-{i}.mp4")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        tmp_path = save_path + ".tmp.mp4"
        save_videos_grid(sample, tmp_path, rescale=True, fps=25)

        command = f'ffmpeg -loglevel quiet -y -i {tmp_path} -i {audio_path} -map 0:v:0 -map 1:a:0 -c:v copy -c:a aac -shortest {save_path}'
        os.system(command)
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    del pipeline
    del transformer3d_val
    del audio2token_val
    del audio_processor
    if not args.disable_mask:
        del face_detector
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()  


def s2v_validation(
    vae, text_encoder, clip_model, net,
    config, args, accelerator, global_step,
    testdata_list, aspect_ratios
):
    transformer3d_val = accelerator.unwrap_model(net).transformer3d
    logging.info("transformer3d_val has been created.")
    
    pipeline = wan.WanS2V(
        config=config,
        checkpoint_dir=args.pretrained_model_name_or_path,
        device=accelerator.device,
        rank=None,
        text_encoder=text_encoder,
        vae=vae,
        model=transformer3d_val,
        validation_mode=True,
    )
    
    for i, (image_path, audio_path, text_prompt) in enumerate(testdata_list):

        with torch.no_grad():
            ref_image = Image.open(image_path).convert("RGB")
            img_height, img_width = int(ref_image.height), int(ref_image.width)
            closest_size, _ = get_closest_ratio(img_height, img_width, ratios=aspect_ratios)
            closest_size = [int(x / 8) * 8 for x in closest_size]

            new_height, new_width = closest_size
            ref_image = ref_image.resize((new_width, new_height), Image.Resampling.LANCZOS)

            if args.motion_frame_num == 0:
                sample = pipeline.generate(
                    input_prompt=text_prompt,
                    ref_images=[ref_image],
                    frame_num=args.video_sample_n_frames,
                    shift=5.0,
                    sample_solver='unipc',
                    sampling_steps=50,
                    offload_model=False,
                )
            else:
                sample = pipeline.generate_motion_frame(
                    input_prompt=text_prompt,
                    ref_images=[ref_image],
                    frame_num=args.video_sample_n_frames,
                    shift=5.0,
                    sample_solver='unipc',
                    sampling_steps=50,
                    offload_model=False,
                    motion_frame_num=args.motion_frame_num,
                )

            sample = sample.unsqueeze(0).cpu()

        save_path = os.path.join(args.output_dir, f"sample/sample-{global_step}-{i}.mp4")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        save_videos_grid(sample, save_path, rescale=True, fps=25)

    del pipeline
    del transformer3d_val
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()  


def log_validation(
    vae, text_encoder, clip_model, net,
    config, args, aspect_ratios, accelerator, global_step
):

    testdata_list = [ [i, a, p] for i, a, p in zip(args.validation_images,
                                                    args.validation_audios, 
                                                    args.validation_prompts)]

    if args.model_type == 'a2v':
        a2v_validation(
            vae, text_encoder, clip_model, net,
            config, args, accelerator, global_step,
            testdata_list, aspect_ratios,
        )

    elif args.model_type in ['s2v', 'fs2v']:
        s2v_validation(
            vae, text_encoder, clip_model, net,
            config, args, accelerator, global_step,
            testdata_list, aspect_ratios,
        )

    else:
        raise TypeError(f"validation not support {args.model_type}")
