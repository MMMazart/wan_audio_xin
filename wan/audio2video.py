# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import gc
import logging
import math
import os
import random
import sys
import types
from contextlib import contextmanager
from functools import partial

import numpy as np
import torch
import torch.cuda.amp as amp
import torch.distributed as dist
import torchvision
import torchvision.transforms.functional as TF
from tqdm import tqdm
from einops import rearrange
from PIL import Image

from .distributed.fsdp import shard_model
from .modules.model import WanModel
from .modules.t5 import T5EncoderModel
from .modules.vae import WanVAE
from .utils.fm_solvers import (FlowDPMSolverMultistepScheduler, get_sampling_sigmas, retrieve_timesteps)
from .utils.fm_solvers_unipc import FlowUniPCMultistepScheduler


class WanA2V:

    def __init__(
        self,
        config,
        checkpoint_dir,
        device_id=0,
        rank=0,
        device=None,
        t5_fsdp=False,
        dit_fsdp=False,
        use_usp=False,
        t5_cpu=False,
        init_on_cpu=True,
        model=None,
        text_encoder=None,
        vae=None,
        audio2token=None,
        validation_mode=False,
    ):
        r"""
        Initializes the image-to-video generation model components.

        Args:
            config (EasyDict):
                Object containing model parameters initialized from config.py
            checkpoint_dir (`str`):
                Path to directory containing model checkpoints
            device_id (`int`,  *optional*, defaults to 0):
                Id of target GPU device
            rank (`int`,  *optional*, defaults to 0):
                Process rank for distributed training
            t5_fsdp (`bool`, *optional*, defaults to False):
                Enable FSDP sharding for T5 model
            dit_fsdp (`bool`, *optional*, defaults to False):
                Enable FSDP sharding for DiT model
            use_usp (`bool`, *optional*, defaults to False):
                Enable distribution strategy of USP.
            t5_cpu (`bool`, *optional*, defaults to False):
                Whether to place T5 model on CPU. Only works without t5_fsdp.
            init_on_cpu (`bool`, *optional*, defaults to True):
                Enable initializing Transformer Model on CPU. Only works without FSDP or USP.
        """
        if device is not None:
            self.device = device
        else:
            self.device = torch.device(f"cuda:{device_id}")
        self.config = config
        self.rank = rank
        self.use_usp = use_usp
        self.t5_cpu = t5_cpu
        self.validation_mode = validation_mode

        self.num_train_timesteps = config.num_train_timesteps
        self.param_dtype = config.param_dtype

        shard_fn = partial(shard_model, device_id=device_id)
        if text_encoder is None:
            self.text_encoder = T5EncoderModel(
                text_len=config.text_len,
                dtype=config.t5_dtype,
                device=torch.device('cpu'),
                checkpoint_path=os.path.join(checkpoint_dir, config.t5_checkpoint),
                tokenizer_path=os.path.join(checkpoint_dir, config.t5_tokenizer),
                shard_fn=shard_fn if t5_fsdp else None,
            )
        else:
            self.text_encoder = text_encoder
        
        self.vae_stride = config.vae_stride
        self.patch_size = config.patch_size
        if vae is None:
            self.vae = WanVAE(
                vae_pth=os.path.join(checkpoint_dir, config.vae_checkpoint),
                device=self.device)
        else:
            self.vae = vae
        
        if model is None:
            logging.info(f"Creating WanModel from {checkpoint_dir}")
            self.model = WanModel.from_pretrained(checkpoint_dir)
            self.model.eval().requires_grad_(False)
        else:
            self.model = model

        self.audio2token = audio2token

        if t5_fsdp or dit_fsdp or use_usp:
            init_on_cpu = False

        if use_usp:
            from xfuser.core.distributed import \
                get_sequence_parallel_world_size

            from .distributed.xdit_context_parallel import (usp_attn_forward,
                                                            usp_dit_forward)
            for block in self.model.blocks:
                block.self_attn.forward = types.MethodType(
                    usp_attn_forward, block.self_attn)
            self.model.forward = types.MethodType(usp_dit_forward, self.model)
            self.sp_size = get_sequence_parallel_world_size()
        else:
            self.sp_size = 1

        if not validation_mode:
            if dist.is_initialized():
                dist.barrier()
        if dit_fsdp:
            self.model = shard_fn(self.model)
        else:
            if not init_on_cpu:
                self.model.to(self.device)

        self.sample_neg_prompt = config.sample_neg_prompt

    
    def get_vae_latents(self, images, device, weight_dtype):
        tensors = []
        for image in images:
            tensors.append(TF.to_tensor(image).sub_(0.5).div_(0.5).to(device))
        tensors = torch.stack(tensors).unsqueeze(0).to(weight_dtype)
        tensors = rearrange(tensors, "b f c h w -> b c f h w")
        latents = self.vae.encode(tensors)
        latents = torch.cat([bs_latents.unsqueeze(0) for bs_latents in latents], dim=0)
        return latents


    def get_audio_token(self, audio_clip, weight_dtype):
        audio_tensor = torch.stack(audio_clip, dim=0).to(device=self.device, dtype=weight_dtype)
        uncond_audio_tensor = torch.zeros_like(audio_tensor)
        audio_tensor = audio_tensor.unsqueeze(0)
        uncond_audio_tensor = uncond_audio_tensor.unsqueeze(0)

        audio_tensor = self.audio2token(audio_tensor)
        uncond_audio_tensor = self.audio2token(uncond_audio_tensor)
        return audio_tensor, uncond_audio_tensor


    def generate(self,
                 input_prompt,
                 ref_images,
                 frame_num=81,
                 shift=5.0,
                 sample_solver='unipc',
                 sampling_steps=50,
                 guide_scale_img=0.0,
                 guide_scale_text=5.0,
                 guide_scale_audio=7.5,
                #  n_prompt='色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走',
                 n_prompt="Blurring, mutation, deformation, distortion, dark and solid, comics, text subtitles, line art, Twisted body, limb deformities, text captions, comic, static, ugly, error, messy code.",
                 seed=42,
                 offload_model=True,
                 audio_list=None,
                 mask=None,
                 ):

        weight_dtype = self.text_encoder.dtype

        ref_latents = self.get_vae_latents(ref_images, self.device, weight_dtype)
        ref_latents_neg = torch.zeros_like(ref_latents)

        F = frame_num
        size = ref_images[0].size
        target_shape = (self.vae.model.z_dim, 
                        (F - 1) // self.vae_stride[0] + 1,
                        size[1] // self.vae_stride[1],
                        size[0] // self.vae_stride[2])

        seq_len = math.ceil((target_shape[2] * target_shape[3]) /
                            (self.patch_size[1] * self.patch_size[2]) *
                            target_shape[1] / self.sp_size) * self.sp_size
        
        seed = seed if seed >= 0 else random.randint(0, sys.maxsize)
        seed_g = torch.Generator(device=self.device)
        seed_g.manual_seed(seed)

        if not self.t5_cpu:
            self.text_encoder.model.to(self.device)
            context = self.text_encoder([input_prompt], self.device)
            context_null = self.text_encoder([n_prompt], self.device)
            if offload_model:
                self.text_encoder.model.cpu()
        else:
            context = self.text_encoder([input_prompt], torch.device('cpu'))
            context_null = self.text_encoder([n_prompt], torch.device('cpu'))
            context = [t.to(self.device) for t in context]
            context_null = [t.to(self.device) for t in context_null]

        audio_list = audio_list[:frame_num]
        audio_tensor, uncond_audio_tensor = self.get_audio_token(audio_list, weight_dtype)

        noise = torch.randn(
            1,
            target_shape[0],
            target_shape[1],
            target_shape[2],
            target_shape[3],
            dtype=weight_dtype,
            device=self.device,
            generator=seed_g)

        @contextmanager
        def noop_no_sync():
            yield

        no_sync = getattr(self.model, 'no_sync', noop_no_sync)

        # evaluation mode
        with amp.autocast(dtype=self.param_dtype), torch.no_grad(), no_sync():

            if sample_solver == 'unipc':
                sample_scheduler = FlowUniPCMultistepScheduler(
                    num_train_timesteps=self.num_train_timesteps,
                    shift=1,
                    use_dynamic_shifting=False)
                sample_scheduler.set_timesteps(
                    sampling_steps, device=self.device, shift=shift)
                timesteps = sample_scheduler.timesteps
            else:
                raise NotImplementedError("Unsupported solver.")

            # sample videos
            latents = noise

            if offload_model:
                torch.cuda.empty_cache()

            if guide_scale_img > 0 and guide_scale_text > 0 and guide_scale_audio > 0:
                # arg_null, arg_i, arg_it, arg_ita
                arg_c = {
                    "context": [context_null[0], context_null[0], context[0], context[0]],
                    "ref_fea": torch.cat([ref_latents_neg, ref_latents, ref_latents, ref_latents], dim=0),
                    "audio_feat": torch.cat([uncond_audio_tensor, uncond_audio_tensor, uncond_audio_tensor, audio_tensor], dim=0),
                    "mask": mask.repeat(4, 1, 1, 1) if mask is not None else None,
                    'seq_len': seq_len,
                }
            elif guide_scale_img == 0 and guide_scale_text > 0 and guide_scale_audio > 0:
                # arg_i, arg_it, arg_ita
                arg_c = {
                    "context": [context_null[0], context[0], context[0]],
                    "ref_fea": torch.cat([ref_latents, ref_latents, ref_latents], dim=0),
                    "audio_feat": torch.cat([uncond_audio_tensor, uncond_audio_tensor, audio_tensor], dim=0),
                    "mask": mask.repeat(3, 1, 1, 1) if mask is not None else None,
                    'seq_len': seq_len,
                }
            elif guide_scale_img == 0 and guide_scale_text == 0 and guide_scale_audio > 0:
                # arg_it, argc_ita
                arg_c = {
                    "context": [context[0], context[0]],
                    "ref_fea": torch.cat([ref_latents, ref_latents], dim=0),
                    "audio_feat": torch.cat([uncond_audio_tensor, audio_tensor], dim=0),
                    "mask": mask.repeat(2, 1, 1, 1) if mask is not None else None,
                    'seq_len': seq_len,
                }
            elif guide_scale_img == 0 and guide_scale_text == 0 and guide_scale_audio == 0:
                # argc_ita
                arg_c = {
                    "context": [context[0]],
                    "ref_fea": torch.cat([ref_latents], dim=0),
                    "audio_feat": torch.cat([audio_tensor], dim=0),
                    "mask": mask.repeat(1, 1, 1, 1) if mask is not None else None,
                    'seq_len': seq_len,
                }
            
            self.model.to(self.device)

            for _, t in enumerate(tqdm(timesteps)):
                latent_model_input = latents.repeat(len(arg_c["context"]), 1, 1, 1, 1)
                timestep = [t]
                timestep = torch.stack(timestep).to(self.device)

                if guide_scale_img > 0 and guide_scale_text > 0 and guide_scale_audio > 0:
                    neg, pos_i, pos_it, pos_ita = self.model(latent_model_input, t=timestep, **arg_c)
                    noise_pred = neg + guide_scale_img * (pos_i - neg) + guide_scale_text * (pos_it - pos_i) + guide_scale_audio * (pos_ita - pos_it)

                elif guide_scale_img == 0 and guide_scale_text > 0 and guide_scale_audio > 0:
                    pos_i, pos_it, pos_ita = self.model(latent_model_input, t=timestep, **arg_c)
                    noise_pred = pos_i + guide_scale_text * (pos_it - pos_i) + guide_scale_audio * (pos_ita - pos_it)

                elif guide_scale_img == 0 and guide_scale_text == 0 and guide_scale_audio > 0:
                    pos_it, pos_ita = self.model(latent_model_input, t=timestep, **arg_c)
                    noise_pred = pos_it + guide_scale_audio * (pos_ita - pos_it)

                elif guide_scale_img == 0 and guide_scale_text == 0 and guide_scale_audio == 0:
                    noise_pred = self.model(latent_model_input, t=timestep, **arg_c)[0]

                temp_x0 = sample_scheduler.step(
                    noise_pred,
                    t,
                    latents,
                    return_dict=False,
                    generator=seed_g)[0]
                
                latents = temp_x0

            x0 = [latents.to(self.device)]

            if offload_model:
                self.model.cpu()
                torch.cuda.empty_cache()

            if self.rank is None or self.rank == 0:
                videos = self.vae.decode(x0[0])

        del noise, latents
        del sample_scheduler
        if offload_model:
            gc.collect()
            torch.cuda.synchronize()
        if not self.validation_mode:
            if dist.is_initialized():
                dist.barrier()
        
        return videos[0]
    

    def generate_motion_frame(self,
                 input_prompt,
                 ref_images,
                 frame_num=81,
                 shift=5.0,
                 sample_solver='unipc',
                 sampling_steps=50,
                 guide_scale_img=0.0,
                 guide_scale_text=5.0,
                 guide_scale_audio=7.5,
                #  n_prompt='色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走',
                 n_prompt="Blurring, mutation, deformation, distortion, dark and solid, comics, text subtitles, line art, Twisted body, limb deformities, text captions, comic, static, ugly, error, messy code.",
                 seed=42,
                 offload_model=True,
                 audio_list=None,
                 mask=None,
                 motion_frame_num=5,
                 ):

        weight_dtype = self.text_encoder.dtype

        ref_latents = self.get_vae_latents(ref_images, self.device, weight_dtype)
        ref_latents_neg = torch.zeros_like(ref_latents)

        motion_frame_latents = self.get_vae_latents(ref_images*motion_frame_num, self.device, weight_dtype)

        F = frame_num
        size = ref_images[0].size
        target_shape = (self.vae.model.z_dim, 
                        (F - 1) // self.vae_stride[0] + 1,
                        size[1] // self.vae_stride[1],
                        size[0] // self.vae_stride[2])

        seq_len = math.ceil((target_shape[2] * target_shape[3]) /
                            (self.patch_size[1] * self.patch_size[2]) *
                            target_shape[1] / self.sp_size) * self.sp_size
        
        seed = seed if seed >= 0 else random.randint(0, sys.maxsize)
        seed_g = torch.Generator(device=self.device)
        seed_g.manual_seed(seed)

        if not self.t5_cpu:
            self.text_encoder.model.to(self.device)
            context = self.text_encoder([input_prompt], self.device)
            context_null = self.text_encoder([n_prompt], self.device)
            if offload_model:
                self.text_encoder.model.cpu()
        else:
            context = self.text_encoder([input_prompt], torch.device('cpu'))
            context_null = self.text_encoder([n_prompt], torch.device('cpu'))
            context = [t.to(self.device) for t in context]
            context_null = [t.to(self.device) for t in context_null]

        audio_list = audio_list[:frame_num]
        audio_tensor, uncond_audio_tensor = self.get_audio_token(audio_list, weight_dtype)

        noise = torch.randn(
            1,
            target_shape[0],
            target_shape[1],
            target_shape[2],
            target_shape[3],
            dtype=weight_dtype,
            device=self.device,
            generator=seed_g)

        @contextmanager
        def noop_no_sync():
            yield

        no_sync = getattr(self.model, 'no_sync', noop_no_sync)

        # evaluation mode
        with amp.autocast(dtype=self.param_dtype), torch.no_grad(), no_sync():

            if sample_solver == 'unipc':
                sample_scheduler = FlowUniPCMultistepScheduler(
                    num_train_timesteps=self.num_train_timesteps,
                    shift=1,
                    use_dynamic_shifting=False)
                sample_scheduler.set_timesteps(
                    sampling_steps, device=self.device, shift=shift)
                timesteps = sample_scheduler.timesteps
            else:
                raise NotImplementedError("Unsupported solver.")

            # sample videos
            latents = noise

            if offload_model:
                torch.cuda.empty_cache()

            if guide_scale_img > 0 and guide_scale_text > 0 and guide_scale_audio > 0:
                # arg_null, arg_i, arg_it, arg_ita
                arg_c = {
                    "context": [context_null[0], context_null[0], context[0], context[0]],
                    "ref_fea": torch.cat([ref_latents_neg, ref_latents, ref_latents, ref_latents], dim=0),
                    "audio_feat": torch.cat([uncond_audio_tensor, uncond_audio_tensor, uncond_audio_tensor, audio_tensor], dim=0),
                    "mask": mask.repeat(4, 1, 1, 1) if mask is not None else None,
                    'motion_frame': torch.cat([motion_frame_latents, motion_frame_latents, motion_frame_latents, motion_frame_latents], dim=0),
                    'seq_len': seq_len,
                }
            elif guide_scale_img == 0 and guide_scale_text > 0 and guide_scale_audio > 0:
                # arg_i, arg_it, arg_ita
                arg_c = {
                    "context": [context_null[0], context[0], context[0]],
                    "ref_fea": torch.cat([ref_latents, ref_latents, ref_latents], dim=0),
                    "audio_feat": torch.cat([uncond_audio_tensor, uncond_audio_tensor, audio_tensor], dim=0),
                    "mask": mask.repeat(3, 1, 1, 1) if mask is not None else None,
                    'motion_frame': torch.cat([motion_frame_latents, motion_frame_latents, motion_frame_latents], dim=0),
                    'seq_len': seq_len,
                }
            elif guide_scale_img == 0 and guide_scale_text == 0 and guide_scale_audio > 0:
                # arg_it, argc_ita
                arg_c = {
                    "context": [context[0], context[0]],
                    "ref_fea": torch.cat([ref_latents, ref_latents], dim=0),
                    "audio_feat": torch.cat([uncond_audio_tensor, audio_tensor], dim=0),
                    "mask": mask.repeat(2, 1, 1, 1) if mask is not None else None,
                    'motion_frame': torch.cat([motion_frame_latents, motion_frame_latents], dim=0),
                    'seq_len': seq_len,
                }
            elif guide_scale_img == 0 and guide_scale_text == 0 and guide_scale_audio == 0:
                # argc_ita
                arg_c = {
                    "context": [context[0]],
                    "ref_fea": torch.cat([ref_latents], dim=0),
                    "audio_feat": torch.cat([audio_tensor], dim=0),
                    "mask": mask.repeat(1, 1, 1, 1) if mask is not None else None,
                    'motion_frame': torch.cat([motion_frame_latents], dim=0),
                    'seq_len': seq_len,
                }

            else:
                raise TypeError("!!!")
            
            self.model.to(self.device)

            for _, t in enumerate(tqdm(timesteps)):
                latent_model_input = latents.repeat(len(arg_c["context"]), 1, 1, 1, 1)
                timestep = [t]
                timestep = torch.stack(timestep).to(self.device)

                if guide_scale_img > 0 and guide_scale_text > 0 and guide_scale_audio > 0:
                    neg, pos_i, pos_it, pos_ita = self.model(latent_model_input, t=timestep, **arg_c)
                    noise_pred = neg + guide_scale_img * (pos_i - neg) + guide_scale_text * (pos_it - pos_i) + guide_scale_audio * (pos_ita - pos_it)

                elif guide_scale_img == 0 and guide_scale_text > 0 and guide_scale_audio > 0:
                    pos_i, pos_it, pos_ita = self.model(latent_model_input, t=timestep, **arg_c)
                    noise_pred = pos_i + guide_scale_text * (pos_it - pos_i) + guide_scale_audio * (pos_ita - pos_it)

                elif guide_scale_img == 0 and guide_scale_text == 0 and guide_scale_audio > 0:
                    pos_it, pos_ita = self.model(latent_model_input, t=timestep, **arg_c)
                    noise_pred = pos_it + guide_scale_audio * (pos_ita - pos_it)

                elif guide_scale_img == 0 and guide_scale_text == 0 and guide_scale_audio == 0:
                    noise_pred = self.model(latent_model_input, t=timestep, **arg_c)[0]

                temp_x0 = sample_scheduler.step(
                    noise_pred,
                    t,
                    latents,
                    return_dict=False,
                    generator=seed_g)[0]
                
                latents = temp_x0

            x0 = [latents.to(self.device)]

            if offload_model:
                self.model.cpu()
                torch.cuda.empty_cache()

            if self.rank is None or self.rank == 0:
                videos = self.vae.decode(x0[0])

        del noise, latents
        del sample_scheduler
        if offload_model:
            gc.collect()
            torch.cuda.synchronize()
        if not self.validation_mode:
            if dist.is_initialized():
                dist.barrier()
        
        return videos[0]
    

    def generate_motion_frame_long(self,
                input_prompt,
                ref_images,
                frame_num=81,
                shift=5.0,
                sample_solver='unipc',
                sampling_steps=50,
                guide_scale_img=0.0,
                guide_scale_text=5.0,
                guide_scale_audio=7.5,
            #  n_prompt='色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走',
                n_prompt="Blurring, mutation, deformation, distortion, dark and solid, comics, text subtitles, line art, Twisted body, limb deformities, text captions, comic, static, ugly, error, messy code.",
                seed=42,
                offload_model=True,
                audio_list=None,
                mask=None,
                motion_frame_num=5,
                ):

        weight_dtype = self.text_encoder.dtype

        ref_latents = self.get_vae_latents(ref_images, self.device, weight_dtype)
        ref_latents_neg = torch.zeros_like(ref_latents)
        _, _, _, lat_h, lat_w = ref_latents.shape

        motion_frames = ref_images * motion_frame_num

        # audio_list pad到frame_num的整数倍
        audio_len = len(audio_list)
        chunk_num = math.ceil(audio_len / frame_num)
        pad_len = chunk_num * frame_num - audio_len
        audio_list = audio_list + [torch.zeros_like(audio_list[0])] * pad_len
        gen_video_list = []
        
        for chunk_idx in range(chunk_num):
            motion_frame_latents = self.get_vae_latents(motion_frames, self.device, weight_dtype)

            lat_f = (frame_num - 1) // self.vae_stride[0] + 1
            seq_len = (lat_f * lat_h * lat_w) // (self.patch_size[1] * self.patch_size[2])
            seq_len = int(math.ceil(seq_len / self.sp_size)) * self.sp_size
        
            seed = seed if seed >= 0 else random.randint(0, sys.maxsize)
            seed_g = torch.Generator(device=self.device)
            seed_g.manual_seed(seed)

            if not self.t5_cpu:
                self.text_encoder.model.to(self.device)
                context = self.text_encoder([input_prompt], self.device)
                context_null = self.text_encoder([n_prompt], self.device)
                if offload_model:
                    self.text_encoder.model.cpu()
            else:
                context = self.text_encoder([input_prompt], torch.device('cpu'))
                context_null = self.text_encoder([n_prompt], torch.device('cpu'))
                context = [t.to(self.device) for t in context]
                context_null = [t.to(self.device) for t in context_null]

            audio_chunk = audio_list[chunk_idx * frame_num : (chunk_idx + 1) * frame_num]
            audio_tensor, uncond_audio_tensor = self.get_audio_token(audio_chunk, weight_dtype)

            noise = torch.randn(
                1,
                self.vae.model.z_dim,
                lat_f,
                lat_h,
                lat_w,
                dtype=weight_dtype,
                device=self.device,
                generator=seed_g)

            @contextmanager
            def noop_no_sync():
                yield

            no_sync = getattr(self.model, 'no_sync', noop_no_sync)

            # evaluation mode
            with amp.autocast(dtype=self.param_dtype), torch.no_grad(), no_sync():

                if sample_solver == 'unipc':
                    sample_scheduler = FlowUniPCMultistepScheduler(
                        num_train_timesteps=self.num_train_timesteps,
                        shift=1,
                        use_dynamic_shifting=False)
                    sample_scheduler.set_timesteps(
                        sampling_steps, device=self.device, shift=shift)
                    timesteps = sample_scheduler.timesteps
                else:
                    raise NotImplementedError("Unsupported solver.")

                # sample videos
                latents = noise

                if offload_model:
                    torch.cuda.empty_cache()

                if guide_scale_img > 0 and guide_scale_text > 0 and guide_scale_audio > 0:
                    # arg_null, arg_i, arg_it, arg_ita
                    arg_c = {
                        "context": [context_null[0], context_null[0], context[0], context[0]],
                        "ref_fea": torch.cat([ref_latents_neg, ref_latents, ref_latents, ref_latents], dim=0),
                        "audio_feat": torch.cat([uncond_audio_tensor, uncond_audio_tensor, uncond_audio_tensor, audio_tensor], dim=0),
                        "mask": mask.repeat(4, 1, 1, 1) if mask is not None else None,
                        'motion_frame': torch.cat([motion_frame_latents, motion_frame_latents, motion_frame_latents, motion_frame_latents], dim=0),
                        'seq_len': seq_len,
                    }
                elif guide_scale_img == 0 and guide_scale_text > 0 and guide_scale_audio > 0:
                    # arg_i, arg_it, arg_ita
                    arg_c = {
                        "context": [context_null[0], context[0], context[0]],
                        "ref_fea": torch.cat([ref_latents, ref_latents, ref_latents], dim=0),
                        "audio_feat": torch.cat([uncond_audio_tensor, uncond_audio_tensor, audio_tensor], dim=0),
                        "mask": mask.repeat(3, 1, 1, 1) if mask is not None else None,
                        'motion_frame': torch.cat([motion_frame_latents, motion_frame_latents, motion_frame_latents], dim=0),
                        'seq_len': seq_len,
                    }
                elif guide_scale_img == 0 and guide_scale_text == 0 and guide_scale_audio > 0:
                    # arg_it, argc_ita
                    arg_c = {
                        "context": [context[0], context[0]],
                        "ref_fea": torch.cat([ref_latents, ref_latents], dim=0),
                        "audio_feat": torch.cat([uncond_audio_tensor, audio_tensor], dim=0),
                        "mask": mask.repeat(2, 1, 1, 1) if mask is not None else None,
                        'motion_frame': torch.cat([motion_frame_latents, motion_frame_latents], dim=0),
                        'seq_len': seq_len,
                    }
                elif guide_scale_img == 0 and guide_scale_text == 0 and guide_scale_audio == 0:
                    # argc_ita
                    arg_c = {
                        "context": [context[0]],
                        "ref_fea": torch.cat([ref_latents], dim=0),
                        "audio_feat": torch.cat([audio_tensor], dim=0),
                        "mask": mask.repeat(1, 1, 1, 1) if mask is not None else None,
                        'motion_frame': torch.cat([motion_frame_latents], dim=0),
                        'seq_len': seq_len,
                    }

                else:
                    raise TypeError("!!!")
                
                self.model.to(self.device)

                for _, t in enumerate(tqdm(timesteps)):
                    latent_model_input = latents.repeat(len(arg_c["context"]), 1, 1, 1, 1)
                    timestep = [t]
                    timestep = torch.stack(timestep)

                    if guide_scale_img > 0 and guide_scale_text > 0 and guide_scale_audio > 0:
                        neg, pos_i, pos_it, pos_ita = self.model(latent_model_input, t=timestep, **arg_c)
                        noise_pred = neg + guide_scale_img * (pos_i - neg) + guide_scale_text * (pos_it - pos_i) + guide_scale_audio * (pos_ita - pos_it)

                    elif guide_scale_img == 0 and guide_scale_text > 0 and guide_scale_audio > 0:
                        pos_i, pos_it, pos_ita = self.model(latent_model_input, t=timestep, **arg_c)
                        noise_pred = pos_i + guide_scale_text * (pos_it - pos_i) + guide_scale_audio * (pos_ita - pos_it)

                    elif guide_scale_img == 0 and guide_scale_text == 0 and guide_scale_audio > 0:
                        pos_it, pos_ita = self.model(latent_model_input, t=timestep, **arg_c)
                        noise_pred = pos_it + guide_scale_audio * (pos_ita - pos_it)

                    elif guide_scale_img == 0 and guide_scale_text == 0 and guide_scale_audio == 0:
                        noise_pred = self.model(latent_model_input, t=timestep, **arg_c)[0]

                    temp_x0 = sample_scheduler.step(
                        noise_pred,
                        t,
                        latents,
                        return_dict=False,
                        generator=seed_g)[0]
                    
                    latents = temp_x0

                x0 = [latents]

                if offload_model:
                    self.model.cpu()
                    torch.cuda.empty_cache()

                if self.rank is None or self.rank == 0:
                    chunk_videos = self.vae.decode(x0[0])
                    chunk_videos = chunk_videos[0].cpu()

                    # to image
                    chunk_videos = chunk_videos.unsqueeze(0)
                    chunk_videos = rearrange(chunk_videos, "b c t h w -> t b c h w")
                    for x in chunk_videos:
                        x = x.float()
                        x = torchvision.utils.make_grid(x, nrow=6)
                        x = x.transpose(0, 1).transpose(1, 2).squeeze(-1)
                        x = (x + 1.0) / 2.0  # -1,1 -> 0,1
                        x = (x * 255).numpy().astype(np.uint8)
                        gen_video_list.append(Image.fromarray(x))
                    motion_frames = gen_video_list[-motion_frame_num:]


        del noise, latents
        del sample_scheduler
        if offload_model:
            gc.collect()
            torch.cuda.synchronize()
        if not self.validation_mode:
            if dist.is_initialized():
                dist.barrier()

        return gen_video_list[:audio_len]
