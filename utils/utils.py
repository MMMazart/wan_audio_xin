import os
import os.path as osp
import glob

from PIL import Image
import cv2
import numpy as np
from einops import rearrange
import imageio
import torch
import torchvision
import torch.nn.functional as F
from diffusers.utils.torch_utils import is_compiled_module


def save_videos_grid(videos: torch.Tensor, path: str, rescale=False, n_rows=6, fps=25, imageio_backend=True, color_transfer_post_process=False):
    def color_transfer(sc, dc):
        """
        Transfer color distribution from of sc, referred to dc.

        Args:
            sc (numpy.ndarray): input image to be transfered.
            dc (numpy.ndarray): reference image

        Returns:
            numpy.ndarray: Transferred color distribution on the sc.
        """

        def get_mean_and_std(img):
            x_mean, x_std = cv2.meanStdDev(img)
            x_mean = np.hstack(np.around(x_mean, 2))
            x_std = np.hstack(np.around(x_std, 2))
            return x_mean, x_std

        sc = cv2.cvtColor(sc, cv2.COLOR_RGB2LAB)
        s_mean, s_std = get_mean_and_std(sc)
        dc = cv2.cvtColor(dc, cv2.COLOR_RGB2LAB)
        t_mean, t_std = get_mean_and_std(dc)
        img_n = ((sc - s_mean) * (t_std / s_std)) + t_mean
        np.putmask(img_n, img_n > 255, 255)
        np.putmask(img_n, img_n < 0, 0)
        dst = cv2.cvtColor(cv2.convertScaleAbs(img_n), cv2.COLOR_LAB2RGB)
        return dst

    videos = rearrange(videos, "b c t h w -> t b c h w")
    outputs = []
    for x in videos:
        x = x.float()
        x = torchvision.utils.make_grid(x, nrow=n_rows)
        x = x.transpose(0, 1).transpose(1, 2).squeeze(-1)
        if rescale:
            x = (x + 1.0) / 2.0  # -1,1 -> 0,1
        x = (x * 255).numpy().astype(np.uint8)
        outputs.append(Image.fromarray(x))

    if color_transfer_post_process:
        for i in range(1, len(outputs)):
            outputs[i] = Image.fromarray(color_transfer(np.uint8(outputs[i]), np.uint8(outputs[0])))

    if os.path.dirname(path) != '':
        os.makedirs(os.path.dirname(path), exist_ok=True)
    if imageio_backend:
        if path.endswith("mp4"):
            imageio.mimsave(path, outputs, fps=fps)
        else:
            imageio.mimsave(path, outputs, duration=(1000 * 1/fps))
    else:
        if path.endswith("mp4"):
            path = path.replace('.mp4', '.gif')
        outputs[0].save(path, format='GIF', append_images=outputs, save_all=True, duration=100, loop=0)


def load_checkpoint(model=None, checkpoint_path=''):
    try:
        if model is not None:
            if checkpoint_path is not None and os.path.exists(checkpoint_path):
                print(f"From checkpoint: {checkpoint_path}")
                if os.path.isfile(checkpoint_path):
                    if checkpoint_path.endswith("safetensors"):
                        from safetensors.torch import load_file, safe_open
                        state_dict = load_file(checkpoint_path)
                    else:
                        state_dict = torch.load(checkpoint_path, map_location="cpu")

                elif os.path.isdir(checkpoint_path):
                    from safetensors.torch import load_file, safe_open
                    model_files_safetensors = glob.glob(os.path.join(checkpoint_path, "*.safetensors"))
                    state_dict = {}
                    for model_file_safetensors in model_files_safetensors:
                        _state_dict = load_file(model_file_safetensors)
                        for key in _state_dict:
                            state_dict[key] = _state_dict[key]

                state_dict = state_dict["state_dict"] if "state_dict" in state_dict else state_dict

                m, u = model.load_state_dict(state_dict, strict=False)
                print(f"missing keys: {len(m)}, unexpected keys: {len(u)}")
            
            else:
                print(f"{checkpoint_path} not exists")

        else:
            print("model is None")
    
    except Exception as e:
        pass


def resume_checkpoint(cfg, save_dir, accelerator):
    """
    Load the most recent checkpoint from the specified directory.

    This function loads the latest checkpoint from the `save_dir` if the `resume_from_checkpoint` parameter is set to "latest".
    If a specific checkpoint is provided in `resume_from_checkpoint`, it loads that checkpoint. If no checkpoint is found,
    it starts training from scratch.

    Args:
        cfg: The configuration object containing training parameters.
        save_dir (str): The directory where checkpoints are saved.
        accelerator: The accelerator object for distributed training.

    Returns:
        int: The global step at which to resume training.
    """
    if cfg.resume_from_checkpoint != "latest":
        resume_dir = cfg.resume_from_checkpoint
    else:
        resume_dir = save_dir
    # Get the most recent checkpoint
    dirs = os.listdir(resume_dir)

    dirs = [d for d in dirs if d.startswith("checkpoint")]
    if len(dirs) > 0:
        dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
        path = dirs[-1]
        accelerator.load_state(os.path.join(resume_dir, path))
        accelerator.print(f"Resuming from checkpoint {path}")
        global_step = int(path.split("-")[1])
    else:
        accelerator.print(
            f"Could not find checkpoint under {resume_dir}, start training from scratch")
        global_step = 0

    return global_step


def save_checkpoint(model: torch.nn.Module, save_dir: str, prefix: str, ckpt_num: int, total_limit: int = -1) -> None:
    """
    Save the model's state_dict to a checkpoint file.

    If `total_limit` is provided, this function will remove the oldest checkpoints
    until the total number of checkpoints is less than the specified limit.

    Args:
        model (nn.Module): The model whose state_dict is to be saved.
        save_dir (str): The directory where the checkpoint will be saved.
        prefix (str): The prefix for the checkpoint file name.
        ckpt_num (int): The checkpoint number to be saved.
        total_limit (int, optional): The maximum number of checkpoints to keep.
            Defaults to None, in which case no checkpoints will be removed.

    Raises:
        FileNotFoundError: If the save directory does not exist.
        ValueError: If the checkpoint number is negative.
        OSError: If there is an error saving the checkpoint.
    """

    if not osp.exists(save_dir):
        raise FileNotFoundError(
            f"The save directory {save_dir} does not exist.")

    if ckpt_num < 0:
        raise ValueError(f"Checkpoint number {ckpt_num} must be non-negative.")

    save_path = osp.join(save_dir, f"{prefix}-{ckpt_num}.pth")

    if total_limit > 0:
        checkpoints = os.listdir(save_dir)
        checkpoints = [d for d in checkpoints if d.startswith(prefix)]
        checkpoints = sorted(
            checkpoints, key=lambda x: int(x.split("-")[1].split(".")[0])
        )

        if len(checkpoints) >= total_limit:
            num_to_remove = len(checkpoints) - total_limit + 1
            removing_checkpoints = checkpoints[0:num_to_remove]
            print(
                f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
            )
            print(
                f"Removing checkpoints: {', '.join(removing_checkpoints)}"
            )

            for removing_checkpoint in removing_checkpoints:
                removing_checkpoint_path = osp.join(
                    save_dir, removing_checkpoint)
                try:
                    os.remove(removing_checkpoint_path)
                except OSError as e:
                    print(
                        f"Error removing checkpoint {removing_checkpoint_path}: {e}")

    state_dict = model.state_dict()
    try:
        torch.save(state_dict, save_path)
        print(f"Checkpoint saved at {save_path}")
    except OSError as e:
        raise OSError(f"Error saving checkpoint at {save_path}: {e}") from e
    

# Function for unwrapping if model was compiled with `torch.compile`.
def unwrap_model(accelerator, model):
    model = accelerator.unwrap_model(model)
    model = model._orig_mod if is_compiled_module(model) else model
    return model


def get_sigmas(accelerator, noise_scheduler, timesteps, n_dim=4, dtype=torch.float32):
    sigmas = noise_scheduler.sigmas.to(device=accelerator.device, dtype=dtype)
    schedule_timesteps = noise_scheduler.timesteps.to(accelerator.device)
    timesteps = timesteps.to(accelerator.device)
    step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

    sigma = sigmas[step_indices].flatten()
    while len(sigma.shape) < n_dim:
        sigma = sigma.unsqueeze(-1)
    return sigma


def custom_mse_loss(noise_pred, target, weighting=None, threshold=50):
    noise_pred = noise_pred.float()
    target = target.float()
    diff = noise_pred - target
    mse_loss = F.mse_loss(noise_pred, target, reduction='none')
    mask = (diff.abs() <= threshold).float()
    masked_loss = mse_loss * mask
    if weighting is not None:
        masked_loss = masked_loss * weighting
    final_loss = masked_loss.mean()
    return final_loss