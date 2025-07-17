import torch
import os
import json
from PIL import Image
import imageio

from wan.modules.audio_proj import AudioProjModel
from wan.modules.t5 import T5EncoderModel
from wan.modules.vae import WanVAE
from wan.configs import WAN_CONFIGS
from wan.modules.model import WanModel
from wan.modules.audio_proj import AudioProjModel
from utils.train_utils import set_vae_device
from utils.utils import save_videos_grid, load_checkpoint
import wan

from a2v_datasets.bucket_sampler import ASPECT_RATIO_512, get_closest_ratio
from tools.audio_processor import AudioProcessor
from tools.face_detect import FaceDetector

import argparse
import time


def parse_args():
    parser = argparse.ArgumentParser(description="wan audio inference")
    # checkpoint path
    parser.add_argument("--task", type=str, default="t2v-1.3B")
    parser.add_argument("--model_type", type=str, default="a2v")
    parser.add_argument("--pretrained_model_name_or_path", type=str, default="/mnt/data/public_ckpt/Wan-AI/Wan2.1-T2V-1.3B")
    parser.add_argument("--transformer_path", type=str, default="")
    parser.add_argument("--audio_pe_path", type=str, default="")
    parser.add_argument("--whisper_path", type=str, default="/mnt/data/public_ckpt/audio2feature/whisper-tiny")
    parser.add_argument("--det_path", type=str, default="/mnt/data/public_ckpt/face_detect/scrfd_10g_shape640x640.onnx")
    # param
    parser.add_argument("--sampling_steps", type=int, default=50, help=f"inference steps")
    parser.add_argument("--sample_guide_scale_img", type=float, default=0.0)
    parser.add_argument("--sample_guide_scale_text", type=float, default=5.0)
    parser.add_argument("--sample_guide_scale_audio", type=float, default=7.5)
    parser.add_argument("--frame_num", type=int, default=81, help=f"frame num")
    parser.add_argument("--video_sample_size", type=int, default=512, help=f"frame num")
    parser.add_argument("--motion_frame_num", type=int, default=5, help=f"motion frame num")
    parser.add_argument("--mode", type=str, default='n81', choices=['n81','motion_frame'])
    parser.add_argument("--multi_face", action="store_true", default=False, help="multi face drive by mask")
    parser.add_argument("--disable_mask", action="store_true", default=False, help="not use face mask oprator in a2v_cross_attn")
    parser.add_argument("--crop", action="store_true", default=False, help="crop image according to sonic preprocess")
    # single test
    parser.add_argument("--image_path", type=str, default='examples/main1.png', help="ref image path if not batch_test")
    parser.add_argument("--audio_path", type=str, default='examples/main1.wav', help="audio path if not batch_test")
    parser.add_argument("--text_prompt", type=str, default='A woman is singing.', help="text prompt if not batch_test")
    parser.add_argument("--save_path", type=str, default='output_dir/out.mp4', help="save path if not batch_test")
    # batch test
    parser.add_argument("--batch_test", action="store_true", help="batch test omnihuman test data")
    parser.add_argument("--batch_test_text_prompt_type", type=str, default="long", choices=['long', 'short', 'empty'])
    parser.add_argument("--save_dir", type=str, default="output_dir/")
    # device
    parser.add_argument("--gpu_id", type=int, default=0)
    args = parser.parse_args()
    return args

args = parse_args()
print(args)

device = f"cuda:{args.gpu_id}"
weight_dtype = torch.bfloat16

if args.batch_test:
    print('batch test')
    testroot = '/mnt/data/1498/test_datasets/test'
    with open(os.path.join(testroot, 'test.json'), 'r') as f:
        items = json.load(f)
    testdata_list = []
    for item in items:
        image_path = os.path.join(testroot, item["image_path"])
        audio_path = os.path.join(testroot, item["audio_path"])
        if args.batch_test_text_prompt_type == 'empty':
            text_prompt = ''
        else:
            text_prompt = item[f'text_{args.batch_test_text_prompt_type}_prompt']
        save_name = os.path.splitext(os.path.basename(item["image_path"]))[0] + ".mp4"
        os.makedirs(args.save_dir, exist_ok=True)
        save_path = os.path.join(args.save_dir, save_name)
        testdata_list.append([image_path, audio_path, text_prompt, save_path])
else:
    print('single test')
    testdata_list = [
        [
            args.image_path, args.audio_path, args.text_prompt, args.save_path
        ]
    ]

aspect_ratios = {key : [x / 512 * args.video_sample_size for x in ASPECT_RATIO_512[key]] for key in ASPECT_RATIO_512.keys()}
audio_processor = AudioProcessor(whisper_path=args.whisper_path)
if not args.disable_mask:
    face_detector = FaceDetector(det_path=args.det_path, gpu_id=args.gpu_id)

wan_config = WAN_CONFIGS[args.task]
text_encoder = T5EncoderModel(
    text_len=wan_config.text_len,
    dtype=wan_config.t5_dtype,
    device=torch.device("cpu"),
    checkpoint_path=os.path.join(args.pretrained_model_name_or_path, wan_config.t5_checkpoint),
    tokenizer_path=os.path.join(args.pretrained_model_name_or_path, wan_config.t5_tokenizer),
    shard_fn=None
)

vae = WanVAE(
    vae_pth=os.path.join(args.pretrained_model_name_or_path, wan_config.vae_checkpoint),    
    device="cpu",
)

transformer3d = WanModel.from_pretrained_a2v(args.pretrained_model_name_or_path, model_type=args.model_type)
audio2token = AudioProjModel(seq_len=10, blocks=5, channels=384, intermediate_dim=1024, output_dim=1536, context_tokens=32)
load_checkpoint(transformer3d, args.transformer_path)
load_checkpoint(audio2token, args.audio_pe_path)

text_encoder.model.to(device)
set_vae_device(vae, device=device, dtype=weight_dtype)
transformer3d = transformer3d.to(device=device, dtype=weight_dtype)
audio2token = audio2token.to(device=device, dtype=weight_dtype)
transformer3d.eval().requires_grad_(False)
audio2token.eval().requires_grad_(False)

pipeline = wan.WanA2V(
    config=wan_config,
    checkpoint_dir=args.pretrained_model_name_or_path,
    device=device,
    rank=None,
    text_encoder=text_encoder,
    vae=vae,
    model=transformer3d,
    audio2token=audio2token,
)


for i, (image_path, audio_path, text_prompt, save_path) in enumerate(testdata_list):
    t0 = time.time()

    with torch.no_grad():
        ref_image = Image.open(image_path).convert("RGB")

        # 线上sonic裁剪逻辑
        if args.crop:
            ref_image = face_detector.sonic_face_crop(ref_image, return_type='PIL')
            if ref_image is None:
                print(f"{image_path} no face detected")
                continue

        img_height, img_width = int(ref_image.height), int(ref_image.width)
        closest_size, _ = get_closest_ratio(img_height, img_width, ratios=aspect_ratios)
        closest_size = [int(x / 8) * 8 for x in closest_size]

        new_height, new_width = closest_size
        ref_image = ref_image.resize((new_width, new_height), Image.Resampling.LANCZOS)

        if args.disable_mask:
            mask = None
        else:
            mask = face_detector.get_face_mask(ref_image, multi_face=args.multi_face)
            if mask is None:
                print(f"{image_path} no face detected")
                continue

        audio_emb = audio_processor.preprocess(audio_path)
        audio_prompts = audio_emb['audio_prompts']
        audio_len = audio_prompts.size(1) // 2

        audio_prompts = torch.cat([torch.zeros_like(audio_prompts[:,:4]), audio_prompts, torch.zeros_like(audio_prompts[:,:6])], 1)
        audio_prompts = audio_prompts.to(device=audio2token.device, dtype=audio2token.dtype)
        
        audio_list = []
        for k in range(audio_len):
            audio_list.append(audio_prompts[:, (k*2):(k*2+10)][0])

        if args.mode == 'n81':
            sample = pipeline.generate(
                input_prompt=text_prompt,
                ref_images=[ref_image],
                frame_num=args.frame_num,
                shift=5.0,
                sample_solver='unipc',
                sampling_steps=args.sampling_steps,
                guide_scale_img=args.sample_guide_scale_img,
                guide_scale_text=args.sample_guide_scale_text,
                guide_scale_audio=args.sample_guide_scale_audio,
                offload_model=False,
                audio_list=audio_list,
                mask=mask,
            )
        elif args.mode == 'motion_frame':
            video_list = pipeline.generate_motion_frame_long(
                input_prompt=text_prompt,
                ref_images=[ref_image],
                frame_num=args.frame_num,
                shift=5.0,
                sample_solver='unipc',
                sampling_steps=args.sampling_steps,
                guide_scale_img=args.sample_guide_scale_img,
                guide_scale_text=args.sample_guide_scale_text,
                guide_scale_audio=args.sample_guide_scale_audio,
                offload_model=False,
                audio_list=audio_list,
                mask=mask,
                motion_frame_num=args.motion_frame_num,
            )

            if os.path.dirname(save_path) != '':
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
            imageio.mimsave(save_path, video_list, fps=25)
            t1 = time.time()
            print(f"save in {save_path}, {t1-t0:.1f}s")
            continue

        sample = sample.unsqueeze(0).cpu()

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        tmp_path = save_path + ".tmp.mp4"
        save_videos_grid(sample, tmp_path, rescale=True, fps=25)

        command = f'ffmpeg -loglevel quiet -y -i {tmp_path} -i {audio_path} -map 0:v:0 -map 1:a:0 -c:v copy -c:a aac -shortest {save_path}'
        os.system(command)
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    t1 = time.time()
    print(f"save in {save_path}, {t1-t0:.1f}s")

