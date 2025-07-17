import gc
import logging
import os
import sys
import random
import numpy as np
import torch
import torchvision
import torch.utils.checkpoint

from einops import rearrange


def encode_prompt(
    text_encoder,
    input_prompt,
    max_length,
    device: torch.device,
    dtype: torch.dtype,
):  
    prompt_embeds, context_lens = text_encoder(input_prompt, device, max_length)
    return prompt_embeds.to(dtype), context_lens

def encode_img_prompt(
    clip_model,
    img_prompt,
    dtype: torch.dtype,
):
    img_prompt = rearrange(img_prompt, "b h w c -> b c 1 h w")  # (-1., 1.)
    # img_prompt = torch.clip((img_prompt / 255. - 0.5) * 2.0, -1.0, 1.0)
    img_prompt_embeds = clip_model.visual(img_prompt.to(dtype))
    return img_prompt_embeds.to(dtype)

# This way is quicker when batch grows up
def batch_encode_vae(vae, pixel_values, device, dtype):  # [C, F, H, W]
    pixel_values = pixel_values.to(dtype=dtype, device=device)
    pixel_values = rearrange(pixel_values, "b f c h w -> b c f h w")
    latents = vae.encode(pixel_values)
    latents = torch.cat([bs_latents.unsqueeze(0) for bs_latents in latents], dim=0)
    return latents.to(dtype)

def set_vae_device(vae, device, dtype=None):
    vae.model.to(device, dtype=dtype)
    vae.mean = vae.mean.to(device, dtype=dtype)
    vae.std = vae.std.to(device, dtype=dtype)
    vae.scale[0] = vae.scale[0].to(device, dtype=dtype)
    vae.scale[1] = vae.scale[1].to(device, dtype=dtype)


EXAMPLE_PROMPT = {
    "t2v-1.3B": {
        "prompt": "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage.",
    },
    "t2v-14B": {
        "prompt": "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage.",
    },
    "t2i-14B": {
        "prompt": "一个朴素端庄的美人",
    },
    "i2v-14B": {
        "prompt":
            "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside.",
        "image":
            "examples/i2v_input.JPG",
    },
}


def linear_decay(initial_value, final_value, total_steps, current_step):
    if current_step >= total_steps:
        return final_value
    current_step = max(0, current_step)
    step_size = (final_value - initial_value) / total_steps
    current_value = initial_value + step_size * current_step
    return current_value

def generate_timestep_with_lognorm(low, high, shape, device="cpu", generator=None):
    u = torch.normal(mean=0.0, std=1.0, size=shape, device=device, generator=generator)
    t = 1 / (1 + torch.exp(-u)) * (high - low) + low
    return torch.clip(t.to(torch.int32), low, high - 1)
