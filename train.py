"""Modified from https://github.com/huggingface/diffusers/blob/main/examples/text_to_image/train_text_to_image.py
"""
#!/usr/bin/env python
# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
import argparse
import logging
import math
import os
import pickle
import shutil
import random
from datetime import datetime
import accelerate
import diffusers
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from packaging import version
from accelerate import Accelerator, DeepSpeedPlugin
from accelerate.logging import get_logger
from accelerate.state import AcceleratorState
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers.optimization import get_scheduler
from diffusers.training_utils import (compute_density_for_timestep_sampling,
                                      compute_loss_weighting_for_sd3)
from omegaconf import OmegaConf
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
from transformers.utils import ContextManagers
import datasets

from a2v_datasets.bucket_sampler import (ASPECT_RATIO_512,
                                    ImageVideoSampler,
                                    AspectRatioBatchImageVideoSampler,
                                    RandomSampler)
from a2v_datasets.dataset_image_video import ImageVideoDataset
from a2v_datasets.collate_fn import CollateFn

from wan.configs import WAN_CONFIGS
from wan.modules.model import WanModel
from wan.modules.t5 import T5EncoderModel
from wan.modules.vae import WanVAE
from wan.modules.audio_proj import AudioProjModel
from wan.utils.fm_solvers import FlowDPMSolverMultistepScheduler
from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
from utils.train_utils import (encode_prompt, 
                               batch_encode_vae, 
                               set_vae_device, 
                               linear_decay)
from utils.utils import (load_checkpoint,
                         resume_checkpoint,
                         unwrap_model,
                         get_sigmas,
                         custom_mse_loss)
from validation import log_validation


PATCH_SIZE = (1, 2, 2)
VAE_STRIDE = (4, 8, 8)
SP_SIZE = 1  # sequence_parallel_world_size

logger = get_logger(__name__, log_level="INFO")


class Net(nn.Module):

    def __init__(
        self,
        args,
        transformer3d,
        audio2token
    ):
        super().__init__()
        self.args = args
        self.transformer3d = transformer3d
        if 'a2v' in self.args.model_type:
            self.audio2token = audio2token
    
    def forward(
        self,
        noisy_latents,
        timesteps,
        context,
        seq_len,
        clip_fea,
        image_latents,
        audio_emb,
        mask,
        ref_fea=None,
        motion_frame=None,
    ):
        
        if 'a2v' in self.args.model_type:
            audio_feat = self.audio2token(audio_emb)
        else:
            audio_feat = None
        
        noise_pred = self.transformer3d(
            noisy_latents,
            timesteps,#.to(noisy_latents.dtype),
            context=context,
            seq_len=seq_len,
            clip_fea=clip_fea,
            y=image_latents,
            mask=mask,
            ref_fea=ref_fea,
            audio_feat=audio_feat,
            motion_frame=motion_frame,
        )
        return noise_pred


def main(args):
    # 1. Config
    logging_dir = os.path.join(args.output_dir, args.logging_dir)
    os.makedirs(logging_dir, exist_ok=True)
    
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, 
                                                      logging_dir=logging_dir)

    deepspeed_plugin = DeepSpeedPlugin(zero_stage=2,
                                       offload_optimizer_device=args.offload_optimizer_device,
                                       gradient_accumulation_steps=1)
    
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        deepspeed_plugin=deepspeed_plugin,
    )
    if accelerator.is_main_process:
        if not args.use_deepspeed and args.report_model_info:
            writer = SummaryWriter(log_dir=logging_dir)

        shutil.copy(args.train_config_path, os.path.join(args.output_dir, "train.yaml"))
    
        # Make one log on every process with the configuration for debugging.
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
            filename=f"{logging_dir}/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_pid{os.getpid()}.log",
        )
        logger.info(accelerator.state, main_process_only=False)
        logging.info(args)

    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()
    
    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)
        torch_rng = torch.Generator(accelerator.device).manual_seed(args.seed + accelerator.process_index)
        batch_sampler_generator = torch.Generator().manual_seed(args.seed)
    else:
        torch_rng = None
        batch_sampler_generator = None
    print(f"Init rng with seed {args.seed + accelerator.process_index}. Process_index is {accelerator.process_index}")

    # For mixed precision training we cast all non-trainable weigths (vae, non-lora text_encoder and non-lora transformer3d) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # 2. Dataset
    aspect_ratios = {key : [x / 512 * args.video_sample_size for x in ASPECT_RATIO_512[key]] for key in ASPECT_RATIO_512.keys()}

    train_dataset = ImageVideoDataset(
        args.train_data_meta, 
        args.train_data_dir,
        video_sample_stride=args.video_sample_stride, 
        video_sample_n_frames=args.video_sample_n_frames, 
        video_repeat=args.video_repeat, 
        text_condition_drop_ratio=args.text_condition_drop_ratio,
        audio_condition_drop_ratio=args.audio_condition_drop_ratio,
        aspect_ratios=aspect_ratios,
        disable_mask=args.disable_mask,
        motion_frame_num=args.motion_frame_num,
    )
    
    if args.enable_bucket:
        batch_sampler = AspectRatioBatchImageVideoSampler(
            sampler=RandomSampler(train_dataset, generator=batch_sampler_generator), 
            dataset=train_dataset.dataset, 
            batch_size=args.train_batch_size, 
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
            persistent_workers=True if args.dataloader_num_workers != 0 else False,
            num_workers=args.dataloader_num_workers,
        )
    else:
        batch_sampler = ImageVideoSampler(RandomSampler(train_dataset, generator=batch_sampler_generator), train_dataset, args.train_batch_size)
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_sampler=batch_sampler, 
            persistent_workers=True if args.dataloader_num_workers != 0 else False,
            num_workers=args.dataloader_num_workers,
        )
    

    # 3. Build model
    def deepspeed_zero_init_disabled_context_manager():
        """
        returns either a context list that includes one that will disable zero.Init or an empty context list
        """
        deepspeed_plugin = AcceleratorState().deepspeed_plugin if accelerate.state.is_initialized() else None
        if deepspeed_plugin is None:
            return []

        return [deepspeed_plugin.zero3_init_context_manager(enable=False)]
    
    # Currently Accelerate doesn't know how to handle multiple models under Deepspeed ZeRO stage 3.
    # For this to work properly all models must be run through `accelerate.prepare`. But accelerate
    # will try to assign the same optimizer with the same weights to all models during
    # `deepspeed.initialize`, which of course doesn't work.
    #
    # For now the following workaround will partially support Deepspeed ZeRO-3, by excluding the 2
    # frozen models from being partitioned during `zero.Init` which gets called during
    # `from_pretrained` So CLIPTextModel and AutoencoderKL will not enjoy the parameter sharding
    # across multiple gpus and only UNet2DConditionModel will get ZeRO sharded.
    wan_config = WAN_CONFIGS[args.task]
    with ContextManagers(deepspeed_zero_init_disabled_context_manager()):
        text_encoder = T5EncoderModel(
            text_len=wan_config.text_len,
            dtype=wan_config.t5_dtype,
            device=torch.device('cpu'),
            checkpoint_path=os.path.join(args.pretrained_model_name_or_path, wan_config.t5_checkpoint),
            tokenizer_path=os.path.join(args.pretrained_model_name_or_path, wan_config.t5_tokenizer),
            shard_fn=None)
        
        vae = WanVAE(
            vae_pth=os.path.join(args.pretrained_model_name_or_path, wan_config.vae_checkpoint),
            device='cpu')
        
        clip_model = None
    
    # Get Transformer
    transformer3d = WanModel.from_pretrained_a2v(args.pretrained_model_name_or_path, model_type=args.model_type).to(weight_dtype)
    audio2token = AudioProjModel(seq_len=10, blocks=5, channels=384, intermediate_dim=1024, output_dim=1536, context_tokens=32).to(weight_dtype)
    
    # Freeze vae and text_encoder and set transformer3d to trainable
    vae.model.requires_grad_(False)
    text_encoder.model.requires_grad_(False)
    transformer3d.requires_grad_(True)
    if clip_model is not None:
        clip_model.model.requires_grad_(False)
    audio2token.requires_grad_(True)
    
    # Load transformer and vae from path if it needs.
    if not args.resume_from_checkpoint:
        load_checkpoint(transformer3d, args.transformer_path)
        load_checkpoint(audio2token, args.audio_pe_path)

    # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:
                models[0].transformer3d.save_pretrained(os.path.join(output_dir, "transformer"), max_shard_size="30GB")
                if 'a2v' in args.model_type:
                    models[0].audio2token.save_pretrained(os.path.join(output_dir, "audio2token"), max_shard_size="30GB")
                if not args.use_deepspeed:
                    weights.pop()

                with open(os.path.join(output_dir, "sampler_pos_start.pkl"), 'wb') as file:
                    pickle.dump([batch_sampler.sampler._pos_start, first_epoch], file)

        def load_model_hook(models, input_dir):
            for i in range(len(models)):
                # pop models so that they are not loaded again
                model = models.pop()

                # load diffusers style into model
                load_model = WanModel.from_pretrained(input_dir, subfolder="transformer")
                model.register_to_config(**load_model.config)

                model.load_state_dict(load_model.state_dict())
                del load_model

            pkl_path = os.path.join(input_dir, "sampler_pos_start.pkl")
            if os.path.exists(pkl_path):
                with open(pkl_path, 'rb') as file:
                    loaded_number, _ = pickle.load(file)
                    batch_sampler.sampler._pos_start = max(loaded_number - args.dataloader_num_workers * accelerator.num_processes * 2, 0)
                print(f"Load pkl from {pkl_path}. Get loaded_number = {loaded_number}.")

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    if args.gradient_checkpointing:
        transformer3d.enable_gradient_checkpointing()
        # net.enable_gradient_checkpointing()
    net = Net(args=args, transformer3d=transformer3d, audio2token=audio2token)

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # A good trainable modules is showed below now.
    if accelerator.is_main_process:
        accelerator.print(f"Trainable modules '{args.trainable_modules}'.")

    # if args.trainable_modules is not None:
    #     for name, module in transformer3d.named_modules():
    #         if any(trainable_mod in name for trainable_mod in args.trainable_modules):
    #             for params in module.parameters():
    #                 params.requires_grad_(True)
    #         else:
    #             for params in module.parameters():
    #                 params.requires_grad_(False)

    # Select trainable params
    trainable_params = list(filter(lambda p: p.requires_grad, net.parameters()))
    trainable_names = [name for name, param in net.named_parameters() if param.requires_grad == True]
    logger.info("\n".join(trainable_names))
    logger.info(f"trainable param number: {len(trainable_names)}")


    # 4. Load optimizer, scheduler, sample solver,
    if args.sample_solver == 'unipc':
        noise_scheduler = FlowUniPCMultistepScheduler(
            num_train_timesteps=1000,
            shift=1,
            use_dynamic_shifting=False)
    
    elif args.sample_solver == 'dpm++':
        noise_scheduler = FlowDPMSolverMultistepScheduler(
            num_train_timesteps=1000,
            shift=1,
            use_dynamic_shifting=False)
    else:
        raise NotImplementedError("Unsupported solver.") 

    # Initialize the optimizer
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )

        optimizer_cls = bnb.optim.AdamW8bit
    elif args.use_came:
        try:
            from came_pytorch import CAME
        except:
            raise ImportError(
                "Please install came_pytorch to use CAME. You can do so by running `pip install came_pytorch`"
            )

        optimizer_cls = CAME
    else:
        optimizer_cls = torch.optim.AdamW

    # Init optimizer
    if args.use_came:
        optimizer = optimizer_cls(
            trainable_params,
            lr=args.learning_rate,
            betas=(0.9, 0.999, 0.9999), 
            eps=(1e-30, 1e-16)
        )
    else:
        optimizer = optimizer_cls(
            trainable_params,
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
        )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
    )
    
    # Prepare everything with our `accelerator`.
    # 这里会转换成bf16
    net, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        net, optimizer, train_dataloader, lr_scheduler
    )
    
    # Move text_encode and vae to gpu and cast to weight_dtype
    set_vae_device(vae, device=accelerator.device, dtype=weight_dtype)
    if clip_model is not None:
        clip_model.model.to(accelerator.device, dtype=weight_dtype)
    text_encoder.model.to(accelerator.device)
    
    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)
    
    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_config = OmegaConf.to_container(args, resolve=True)
        tracker_config.pop("validation_images")
        tracker_config.pop("validation_audios")
        tracker_config.pop("validation_prompts")
        tracker_config.pop("trainable_modules")
        accelerator.init_trackers(args.tracker_project_name, tracker_config)
    

    # 5. Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        logger.info(f"Loading checkpoint from {args.output_dir}")
        global_step = resume_checkpoint(args, args.output_dir, accelerator)
        first_epoch = global_step // num_update_steps_per_epoch

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    # ************************ pre validation **************************
    if accelerator.is_main_process and args.pre_validation:
            log_validation(
                vae,
                text_encoder,
                clip_model,
                net,
                wan_config,
                args,
                aspect_ratios,
                accelerator,
                global_step,
            )
    # ************************ pre validation **************************
    for epoch in range(first_epoch, args.num_train_epochs):
        train_loss = 0.0
        batch_sampler.sampler.generator = torch.Generator().manual_seed(args.seed + epoch)
        for step, batch in enumerate(train_dataloader):
            if batch is None:
                continue

            with accelerator.accumulate(net):
                video_pixel_values = batch["video_pixel_values"].to(weight_dtype)
                ref_pixel_values = batch["ref_pixel_values"].to(weight_dtype)
                audio_values = batch["audio_values"].to(weight_dtype)
                mask = batch["mask"]
                if mask is not None:
                    mask = mask.to(weight_dtype)
                motion_frame_pixel_values = batch["motion_frame_pixel_values"]
                if motion_frame_pixel_values is not None:
                    motion_frame_pixel_values = motion_frame_pixel_values.to(weight_dtype)
                
                # Convert images to latent space
                with torch.no_grad():
                    # bz * c * f * h * w
                    latents = batch_encode_vae(vae, video_pixel_values, accelerator.device, weight_dtype)                        
                    ref_latents = batch_encode_vae(vae, ref_pixel_values, accelerator.device, weight_dtype)
                    if motion_frame_pixel_values is not None:
                        motion_frame_latents = batch_encode_vae(vae, motion_frame_pixel_values, accelerator.device, weight_dtype)
                    else:
                        motion_frame_latents = None
                    
                    if args.ref_condition_drop_ratio > 0:
                        for i in range(ref_latents.shape[0]):
                            if random.random() < args.ref_condition_drop_ratio:
                                ref_latents[i] *= 0
                
                # Get text embeds
                with torch.no_grad():
                    prompt_embeds, context_lens = encode_prompt(
                            text_encoder,
                            batch['text'],
                            args.tokenizer_max_length,
                            latents.device,
                            weight_dtype
                        )

                # Get Noise
                bsz = latents.shape[0]
                if args.noise_share_in_frames:
                    def generate_noise(bs, channel, length, height, width, ratio=0.5, generator=None, device="cuda", dtype=None):
                        noise = torch.randn(bs, channel, length, height, width, generator=generator, device=device, dtype=dtype)
                        for i in range(1, length):
                            noise[:, :, i, :, :] = ratio * noise[:, :, i - 1, :, :] + (1 - ratio) * noise[:, :, i, :, :]
                        return noise
                    noise = generate_noise(*latents.size(), ratio=args.noise_share_in_frames_ratio, device=latents.device, generator=torch_rng, dtype=weight_dtype)
                else:
                    noise = torch.randn(latents.size(), device=latents.device, generator=torch_rng, dtype=weight_dtype)
                
                _, nf, _, height, width = batch["video_pixel_values"].shape

                if accelerator.is_main_process and global_step % 100 == 0:
                    logging.info(f"video size: {nf} x {height} x {width}")
                
                # To latents.device
                prompt_embeds = prompt_embeds.to(device=latents.device)

                u = compute_density_for_timestep_sampling(
                    weighting_scheme=args.weighting_scheme,
                    batch_size=bsz,
                    logit_mean=args.logit_mean,
                    logit_std=args.logit_std,
                    mode_scale=args.mode_scale,
                )
                indices = (u * noise_scheduler.config.num_train_timesteps).long()
                timesteps = noise_scheduler.timesteps[indices].to(device=latents.device)

                # Add noise according to flow matching.
                # zt = (1 - texp) * x + texp * z1
                sigmas = get_sigmas(accelerator, noise_scheduler, timesteps, n_dim=latents.ndim, dtype=latents.dtype)
                noisy_latents = (1.0 - sigmas) * latents + sigmas * noise
                
                # Add noise
                target = noise - latents

                target_shape = (vae.model.z_dim, (nf - 1) // 4 + 1, height // 8, width // 8)
                seq_len = math.ceil((target_shape[2] * target_shape[3]) / (PATCH_SIZE[1] * PATCH_SIZE[2]) * target_shape[1] / SP_SIZE) * SP_SIZE

                # Predict the noise residual
                noise_pred = net(
                    noisy_latents,
                    timesteps.to(noisy_latents.dtype),
                    context=prompt_embeds,
                    seq_len=seq_len,
                    clip_fea=None,
                    image_latents=None,
                    audio_emb=audio_values,
                    mask=mask,
                    ref_fea=ref_latents,
                    motion_frame=motion_frame_latents,
                )
                
                noise_pred = torch.cat([noise_pred_batch.unsqueeze(0) for noise_pred_batch in noise_pred], dim=0)
                
                if noise_pred.size()[1] != vae.model.z_dim:
                    noise_pred, _ = noise_pred.chunk(2, dim=1)
                
                if args.loss_type == "ddpm":
                    loss = custom_mse_loss(noise_pred.float(), target.float())
                else:
                    weighting = compute_loss_weighting_for_sd3(weighting_scheme=args.weighting_scheme, sigmas=sigmas)
                    loss = custom_mse_loss(noise_pred.float(), target.float(), weighting.float())
                    loss = loss.mean()

                if args.motion_sub_loss and noise_pred.size()[2] > 2:
                    gt_sub_noise = noise_pred[:, :, 1:].float() - noise_pred[:, :, :-1].float()
                    pre_sub_noise = target[:, :, 1:].float() - target[:, :, :-1].float()
                    sub_loss = F.mse_loss(gt_sub_noise, pre_sub_noise, reduction="mean")
                    loss = loss * (1 - args.motion_sub_loss_ratio) + sub_loss * args.motion_sub_loss_ratio


                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps

                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    if not args.use_deepspeed:
                        trainable_params_grads = [p.grad for p in trainable_params if p.grad is not None]
                        trainable_params_total_norm = torch.norm(torch.stack([torch.norm(g.detach(), 2) for g in trainable_params_grads]), 2)
                        max_grad_norm = linear_decay(args.max_grad_norm * args.initial_grad_norm_ratio, args.max_grad_norm, args.abnormal_norm_clip_start, global_step)
                        if trainable_params_total_norm / max_grad_norm > 5 and global_step > args.abnormal_norm_clip_start:
                            actual_max_grad_norm = max_grad_norm / min((trainable_params_total_norm / max_grad_norm), 10)
                        else:
                            actual_max_grad_norm = max_grad_norm
                    else:
                        actual_max_grad_norm = args.max_grad_norm

                    if not args.use_deepspeed and args.report_model_info and accelerator.is_main_process:
                        if trainable_params_total_norm > 1 and global_step > args.abnormal_norm_clip_start:
                            for name, param in transformer3d.named_parameters():
                                if param.requires_grad:
                                    writer.add_scalar(f'gradients/before_clip_norm/{name}', param.grad.norm(), global_step=global_step)

                    norm_sum = accelerator.clip_grad_norm_(trainable_params, actual_max_grad_norm)
                    if not args.use_deepspeed and args.report_model_info and accelerator.is_main_process:
                        writer.add_scalar(f'gradients/norm_sum', norm_sum, global_step=global_step)
                        writer.add_scalar(f'gradients/actual_max_grad_norm', actual_max_grad_norm, global_step=global_step)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:

                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0

                if global_step % args.checkpointing_steps == 0:
                    if args.use_deepspeed or accelerator.is_main_process:
                        # # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        # if args.checkpoints_total_limit is not None:
                        #     checkpoints = os.listdir(args.output_dir)
                        #     checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                        #     checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                        #     # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                        #     if len(checkpoints) >= args.checkpoints_total_limit:
                        #         num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                        #         removing_checkpoints = checkpoints[0:num_to_remove]

                        #         logger.info(
                        #             f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                        #         )
                        #         logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                        #         for removing_checkpoint in removing_checkpoints:
                        #             removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                        #             shutil.rmtree(removing_checkpoint)

                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")


                if accelerator.is_main_process:
                    if global_step % args.validation_steps == 0:
                        log_validation(
                            vae,
                            text_encoder,
                            clip_model,
                            net,
                            wan_config,
                            args,
                            aspect_ratios,
                            accelerator,
                            global_step,
                        )
            
            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            logger.info(f"global_step: {global_step}, step_loss: {loss.detach().item()}, lr: {lr_scheduler.get_last_lr()[0]}")

            if global_step >= args.max_train_steps:
                break

    # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        transformer3d = unwrap_model(accelerator, transformer3d)

        if args.use_deepspeed or accelerator.is_main_process:
            save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
            accelerator.save_state(save_path)
            logger.info(f"Saved state to {save_path}")

    accelerator.end_training()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./config/train_wan_a2v.yaml")
    args = parser.parse_args()
    config = OmegaConf.load(args.config)
    config.train_config_path = args.config
    
    main(config)
