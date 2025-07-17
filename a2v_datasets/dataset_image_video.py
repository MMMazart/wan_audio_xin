import gc
import json
import os
import random
from contextlib import contextmanager

import cv2
import numpy as np
import torch
from decord import VideoReader
from func_timeout import FunctionTimedOut, func_timeout
from torch.utils.data.dataset import Dataset
from transformers import CLIPImageProcessor
import pickle
import albumentations as A

VIDEO_READER_TIMEOUT = 20


@contextmanager
def VideoReader_contextmanager(*args, **kwargs):
    vr = VideoReader(*args, **kwargs)
    try:
        yield vr
    finally:
        del vr
        gc.collect()

def get_video_reader_batch(video_reader, batch_index):
    frames = video_reader.get_batch(batch_index).asnumpy()
    return frames

def resize_frame(frame, closest_size):
    new_h, new_w = closest_size
    resized_frame = cv2.resize(frame, (new_w, new_h))
    return resized_frame

def get_closest_ratio(height, width, ratios):
    aspect_ratio = height / width
    closest_ratio = min(ratios.keys(), key=lambda ratio: abs(float(ratio) - aspect_ratio))
    return ratios[closest_ratio], float(closest_ratio)


class MotionFrameAugmentation:
    def __init__(self, p=1):
        self.transform = A.SomeOf([
            A.GaussianBlur(blur_limit=(3,5), p=1),
            A.MedianBlur(blur_limit=(3,5), p=1),
            A.Blur(blur_limit=(3,5), p=1),
            A.ImageCompression(quality_lower=50, quality_upper=80, p=1),
            A.Downscale(scale_min=0.8, scale_max=0.9, p=1),
            # A.GaussNoise(var_limit=(0.001, 0.002), p=1),    # 加噪声 暂不使用
            # A.CoarseDropout(max_holes=50, max_height=10, max_width=10, fill_value=0, p=1),   # 遮挡 暂不使用
        ], n=1, p=p)

    def __call__(self, img):
        aug_img = self.transform(image=img)['image']
        # cv2.imwrite("aug.jpg", np.concatenate([img, aug_img], axis=1)[:,:,::-1])
        return aug_img


class ImageVideoDataset(Dataset):
    def __init__(
            self,
            ann_path, 
            data_root=None,
            video_sample_stride=1,
            video_sample_n_frames=81,
            video_repeat=1,
            text_condition_drop_ratio=0.1,
            audio_condition_drop_ratio=0.1,
            video_length_drop_start=0.0, 
            video_length_drop_end=1.0,
            aspect_ratios=None,
            disable_mask=True,
            motion_frame_num=0,
        ):
        self.data_root = data_root

        # read dataset
        meta_file_paths = ann_path.split(',')
        self.dataset = []
        for meta_file_path in meta_file_paths:
            if meta_file_path.endswith('.json') and os.path.exists(meta_file_path):
                with open(meta_file_path, 'r', encoding='utf-8') as f:
                    items = json.load(f)
                filter_items = []
                for item in items:
                    if item['type'] != 'video':
                        continue

                    filter_items.append(item)
                    if item['video_with_hand'] == True or item['body_type'] in ['upper_body', 'full_body']: # 'unkown'、'head'、'upper_body'、'full_body'
                        for _ in range(1):
                            filter_items.append(item)

                print(f"loading annotations from {meta_file_path}, {len(filter_items)} files.")
                self.dataset += filter_items
        
        print(f"dataset scale: {len(self.dataset)} x {int(video_repeat)} = {len(self.dataset) * int(video_repeat)} files")
        self.dataset *= int(video_repeat)

        # # condition params
        self.text_condition_drop_ratio = text_condition_drop_ratio
        self.audio_condition_drop_ratio = audio_condition_drop_ratio

        # # video params
        self.video_length_drop_start = video_length_drop_start
        self.video_length_drop_end = video_length_drop_end

        self.video_sample_stride    = video_sample_stride
        self.video_sample_n_frames  = video_sample_n_frames

        # self.clip_image_processor = CLIPImageProcessor()
        self.aspect_ratios = aspect_ratios
        self.disable_mask = disable_mask
        self.motion_frame_num = motion_frame_num
        if self.motion_frame_num > 0:
            self.aug = MotionFrameAugmentation(p=0.6)


    def draw_mask_by_box(self, info, closest_size, mode='face'):
        assert mode in ['face', 'body']
        x1, y1, x2, y2 = info[f'union_{mode}box']
        mask = np.zeros((info['height'], info['width']), dtype=np.uint8)
        cv2.rectangle(mask, (int(x1), int(y1)), (int(x2), int(y2)), 255, -1)
        mask = resize_frame(mask, closest_size)
        mask = (mask > 127).astype('uint8') * 255
        return mask
    
    def get_audio_emb(self, info):
        audio_emb_path = info['whisper_path']
        if self.data_root is not None:
            audio_emb_path = os.path.join(self.data_root, audio_emb_path)

        with open(audio_emb_path, 'rb') as f:
            audio_emb = pickle.load(f)
        assert "audio_prompts" in audio_emb
        audio_prompts = audio_emb["audio_prompts"]
        if isinstance(audio_prompts, np.ndarray):
            audio_prompts = torch.from_numpy(audio_prompts)
        return audio_prompts

    def get_sample(self, idx):
        data_info = self.dataset[idx % len(self.dataset)]

        closest_size, _ = get_closest_ratio(data_info["height"], data_info["width"], ratios=self.aspect_ratios)
        closest_size = [int(x / 16) * 16 for x in closest_size]

        assert 'text' in data_info and data_info['text'] is not None
        text = data_info['text']

        # fmask
        if not self.disable_mask:
            mask = self.draw_mask_by_box(data_info, closest_size, 'face')
            mask = mask[None]
        else:
            mask = None

        # audio prompt
        audio_prompts = self.get_audio_emb(data_info)
        audio_len = audio_prompts.shape[1] // 2

        video_path = data_info['video_path']
        if self.data_root is not None:
            video_path = os.path.join(self.data_root, video_path)

        with VideoReader_contextmanager(video_path, num_threads=2) as video_reader:
            min_sample_n_frames = min(
                self.video_sample_n_frames, 
                int(len(video_reader) * (self.video_length_drop_end - self.video_length_drop_start) // self.video_sample_stride),
                int(audio_len * (self.video_length_drop_end - self.video_length_drop_start) // self.video_sample_stride),
            )
            if min_sample_n_frames == 0:
                raise ValueError(f"No Frames in video.")
            
            video_length = min(int(self.video_length_drop_end * len(video_reader)), int(self.video_length_drop_end * audio_len))
            clip_length = int(min(video_length, (min_sample_n_frames - 1) * self.video_sample_stride + 1))
            start_idx   = random.randint(int(self.video_length_drop_start * video_length), video_length - clip_length) if video_length != clip_length else 0
            video_idx_list = np.linspace(start_idx, start_idx + clip_length - 1, min_sample_n_frames, dtype=int)

            try:
                # audio tensor
                audio_prompts = torch.cat([torch.zeros_like(audio_prompts[:, :4]), audio_prompts, torch.zeros_like(audio_prompts[:, :6])], dim=1)
                audio_clip = []
                for i in video_idx_list:
                    audio_clip.append(audio_prompts[:, (i*2):(i*2+10)][0])
                assert len(audio_clip) == min_sample_n_frames, "audio clip len != min_sample_n_frames"
                audio_tensor = torch.stack(audio_clip)      # [f,10,5,384]

                # video images
                video_pixel_values = func_timeout(
                    VIDEO_READER_TIMEOUT, get_video_reader_batch, args=(video_reader, video_idx_list)
                )
                video_pixel_values = [resize_frame(frame, closest_size) for frame in video_pixel_values]
                
                # ref images
                ref_idx_list = [random.randint(0, len(video_reader)-1)]

                ref_pixel_values = func_timeout(
                    VIDEO_READER_TIMEOUT, get_video_reader_batch, args=(video_reader, ref_idx_list)
                )
                ref_pixel_values = [resize_frame(frame, closest_size) for frame in ref_pixel_values]

                # motion frames
                if self.motion_frame_num > 0:
                    if random.random() < 0.1:
                        motion_frame_idx_list = [max(0, start_idx - 1)] * self.motion_frame_num    # 情况一：前一帧copy n份，和第一个chunk推理对齐
                    else:
                        motion_frame_idx_list = [max(0, i) for i in range(start_idx - self.motion_frame_num, start_idx)]   # 情况二：前n帧

                    motion_frame_pixel_values = func_timeout(
                        VIDEO_READER_TIMEOUT, get_video_reader_batch, args=(video_reader, motion_frame_idx_list)
                    )
                    motion_frame_pixel_values = [self.aug(resize_frame(frame, closest_size)) for frame in motion_frame_pixel_values]
                else:
                    motion_frame_pixel_values = None

            except FunctionTimedOut:
                raise ValueError(f"Read {idx} timeout.")
            except Exception as e:
                raise ValueError(f"Failed to extract frames from video. Error is {e}.")
            
            # Random use no text generation
            if random.random() < self.text_condition_drop_ratio:
                text = ''

            # Random use no audio generation
            if random.random() < self.audio_condition_drop_ratio:
                audio_tensor = torch.zeros_like(audio_tensor)

        return {
            "video_pixel_values": video_pixel_values,
            "ref_pixel_values": ref_pixel_values,
            "motion_frame_pixel_values": motion_frame_pixel_values,
            "audio_tensor": audio_tensor,
            "mask": mask,
            "text": text,
            "data_type": "video",
            "idx": idx,
            "video_path": video_path,
        }
    
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        while True:
            sample = {}
            try:
                sample = self.get_sample(idx)
                return sample

            except Exception as e:
                print(e, self.dataset[idx % len(self.dataset)])
                idx = random.randint(0, len(self.dataset)-1)



if __name__ == "__main__":
    from tqdm import tqdm

    import sys, os
    sys.path.append(os.getcwd())
    from omegaconf import OmegaConf
    from a2v_datasets.bucket_sampler import (ASPECT_RATIO_512,
                                    AspectRatioBatchImageVideoSampler,
                                    RandomSampler)
    from a2v_datasets.collate_fn import CollateFn

    batch_sampler_generator = torch.Generator().manual_seed(42)
    args = OmegaConf.load('config/train_wan_a2v.yaml')
    aspect_ratios = {key : [x / 512 * args.video_sample_size for x in ASPECT_RATIO_512[key]] for key in ASPECT_RATIO_512.keys()}

    train_dataset = ImageVideoDataset(
        data_root='/mnt/data/1498',
        ann_path="/mnt/data/1498/a2v_youtube_solosing_0325.json,",
        aspect_ratios=aspect_ratios,
        disable_mask=False,
        motion_frame_num=5,
    )

    # for sample in tqdm(train_dataset):
    #     pass


    batch_sampler = AspectRatioBatchImageVideoSampler(
        sampler=RandomSampler(train_dataset, generator=batch_sampler_generator), 
        dataset=train_dataset.dataset, 
        batch_size=4, 
        train_folder = args.train_data_dir, 
        drop_last=True,
        aspect_ratios=aspect_ratios,
    )

    collate_fn = CollateFn(
        args=args,
        aspect_ratios=aspect_ratios,
        sample_n_frames_bucket_interval=4,
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_sampler=batch_sampler,
        collate_fn=collate_fn,
        persistent_workers=False,
        num_workers=0,
    )
    
    from wan.modules.audio_proj import AudioProjModel
    audio2token = AudioProjModel(seq_len=10, blocks=5, channels=384, intermediate_dim=1024, output_dim=1536, context_tokens=32)
    
    for step, batch in enumerate(tqdm(train_dataloader)):
        audio_values = batch["audio_values"]    # [bs, n, 10, 5, 384]
        audio_feat = audio2token(audio_values)  # [bs, n, 32, 1536]

        break

