import torch
import os
import json
from PIL import Image
import imageio

from wan.modules.t5 import T5EncoderModel
from wan.modules.vae import WanVAE
from wan.configs import WAN_CONFIGS
from wan.modules.model import WanModel
from utils.train_utils import set_vae_device
from utils.utils import save_videos_grid, load_checkpoint
import wan

from a2v_datasets.bucket_sampler import ASPECT_RATIO_512, get_closest_ratio

import argparse
import time


def parse_args():
    parser = argparse.ArgumentParser(description="wan audio inference")
    # checkpoint path
    parser.add_argument("--task", type=str, default="t2v-1.3B")
    parser.add_argument("--model_type", type=str, default="s2v")
    parser.add_argument("--pretrained_model_name_or_path", type=str, default="/mnt/data/public_ckpt/Wan-AI/Wan2.1-T2V-1.3B")
    parser.add_argument("--transformer_path", type=str, default="checkpoints/20250711_fs2v_wo-dropout/checkpoint-12000/transformer/diffusion_pytorch_model.safetensors")
    # param
    parser.add_argument("--sampling_steps", type=int, default=50, help=f"inference steps")
    parser.add_argument("--sample_guide_scale_img", type=float, default=0.0)
    parser.add_argument("--sample_guide_scale_text", type=float, default=5.0)
    parser.add_argument("--frame_num", type=int, default=81, help=f"frame num")
    parser.add_argument("--video_sample_size", type=int, default=512, help=f"frame num")
    parser.add_argument("--motion_frame_num", type=int, default=5, help=f"motion frame num")
    parser.add_argument("--mode", type=str, default='n81', choices=['n81', 'sonic', 'overlap', 'motion_frame'])
    # single test
    parser.add_argument("--image_path", type=str, default='examples/main1.png', help="ref image path if not batch_test")
    parser.add_argument("--text_prompt", type=str, default='A woman is singing and playing guitar.', help="text prompt if not batch_test")
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
        if args.batch_test_text_prompt_type == 'empty':
            text_prompt = ''
        else:
            text_prompt = item[f'text_{args.batch_test_text_prompt_type}_prompt']
        save_name = os.path.splitext(os.path.basename(item["image_path"]))[0] + ".mp4"
        os.makedirs(args.save_dir, exist_ok=True)
        save_path = os.path.join(args.save_dir, save_name)
        testdata_list.append([image_path, text_prompt, save_path])
else:
    print('single test')
    testdata_list = [
        [
            args.image_path, args.text_prompt, args.save_path
        ]
    ]


aspect_ratios = {key : [x / 512 * args.video_sample_size for x in ASPECT_RATIO_512[key]] for key in ASPECT_RATIO_512.keys()}

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
    device="cpu"
)

transformer3d = WanModel.from_pretrained_a2v(args.pretrained_model_name_or_path, model_type=args.model_type)
load_checkpoint(transformer3d, args.transformer_path)

set_vae_device(vae, device=device, dtype=weight_dtype)
transformer3d = transformer3d.to(device=device, dtype=weight_dtype)
transformer3d.eval().requires_grad_(False)

pipeline = wan.WanS2V(
    config=wan_config,
    checkpoint_dir=args.pretrained_model_name_or_path,
    device=device,
    rank=None,
    text_encoder=text_encoder,
    vae=vae,
    model=transformer3d,
)


for image_path, text_prompt, save_path in testdata_list:
    t0 = time.time()

    with torch.no_grad():
        ref_image = Image.open(image_path).convert("RGB")
        img_height, img_width = int(ref_image.height), int(ref_image.width)
        closest_size, _ = get_closest_ratio(img_height, img_width, ratios=aspect_ratios)
        closest_size = [int(x / 8) * 8 for x in closest_size]

        new_height, new_width = closest_size
        ref_image = ref_image.resize((new_width, new_height), Image.Resampling.LANCZOS)

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
                offload_model=False,
            )
        elif args.mode == 'sonic':
            sample = pipeline.generate_sonic(
                input_prompt=text_prompt,
                ref_images=[ref_image],
                frame_num=args.frame_num,
                shift=5.0,
                sample_solver='unipc',
                sampling_steps=args.sampling_steps,
                guide_scale_img=args.sample_guide_scale_img,
                guide_scale_text=args.sample_guide_scale_text,
                offload_model=False,
            )
        elif args.mode == 'overlap':
            sample = pipeline.generate_overlap(
                input_prompt=text_prompt,
                ref_images=[ref_image],
                frame_num=args.frame_num,
                shift=5.0,
                sample_solver='unipc',
                sampling_steps=args.sampling_steps,
                guide_scale_img=args.sample_guide_scale_img,
                guide_scale_text=args.sample_guide_scale_text,
                offload_model=False,
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
                offload_model=False,
                motion_frame_num=args.motion_frame_num,
            )

            if os.path.dirname(save_path) != '':
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
            imageio.mimsave(save_path, video_list, fps=25)
            t1 = time.time()
            print(f"save in {save_path}, {t1-t0:.1f}s")
            continue

        sample = sample.unsqueeze(0).cpu()

        if os.path.dirname(save_path) != '':
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
        save_videos_grid(sample, save_path, rescale=True, fps=25)

    t1 = time.time()
    print(f"save in {save_path}, {t1-t0:.1f}s")
