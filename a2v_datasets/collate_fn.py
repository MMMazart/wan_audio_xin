import random

import numpy as np
import torch
from torchvision import transforms
import torchvision.transforms.functional as TF


class CollateFn:
    def __init__(self, 
                 args, 
                 aspect_ratios,
                 sample_n_frames_bucket_interval=4,
                 ):
        self.args = args
        self.sample_n_frames_bucket_interval = sample_n_frames_bucket_interval
        self.aspect_ratios = aspect_ratios

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
        ])
        # img = TF.to_tensor(img).sub_(0.5).div_(0.5)
        # img = self.transform(img)


    def __call__(self, examples):
        try:
            args = self.args
            sample_n_frames_bucket_interval = self.sample_n_frames_bucket_interval

            # Create new output
            new_examples                 = {}
            new_examples["video_pixel_values"] = []
            new_examples["ref_pixel_values"] = []
            new_examples["text"]         = []
            new_examples["audio_values"] = []
            new_examples["clip_pixel_values"] = []
            
            # mask
            if examples[0]["mask"] is None:
                new_examples["mask"] = None
            else:
                new_examples["mask"] = []

            # motion frames
            if examples[0]['motion_frame_pixel_values'] is None:
                new_examples["motion_frame_pixel_values"] = None
            else:
                new_examples["motion_frame_pixel_values"] = []

            
            batch_video_length = args.video_sample_n_frames + sample_n_frames_bucket_interval

            for example in examples:
                # video
                video_pixel_values = [self.transform(img) for img in example["video_pixel_values"]]
                video_pixel_values = torch.from_numpy(np.array(video_pixel_values))
                new_examples["video_pixel_values"].append(video_pixel_values)

                # ref
                ref_pixel_values = [self.transform(img) for img in example["ref_pixel_values"]]
                ref_pixel_values = torch.from_numpy(np.array(ref_pixel_values))
                new_examples["ref_pixel_values"].append(ref_pixel_values)

                # mask
                if new_examples["mask"] is not None:
                    mask = torch.from_numpy(example["mask"])
                    mask = mask / 255.
                    new_examples["mask"].append(mask)

                # motion frame
                if new_examples["motion_frame_pixel_values"] is not None:
                    motion_frame_pixel_values = [self.transform(img) for img in example["motion_frame_pixel_values"]]
                    motion_frame_pixel_values = torch.from_numpy(np.array(motion_frame_pixel_values))
                    new_examples["motion_frame_pixel_values"].append(motion_frame_pixel_values)

                # text
                new_examples["text"].append(example["text"])

                # audio
                new_examples["audio_values"].append(example["audio_tensor"])

                
                # needs the number of frames to be 4n + 1.
                batch_video_length = int(
                    min(
                        batch_video_length,
                        (len(video_pixel_values) - 1) // sample_n_frames_bucket_interval * sample_n_frames_bucket_interval + 1, 
                    )
                )
                
                if batch_video_length == 0:
                    batch_video_length = 1
                
                # CLIP Image for vision ipadapter
                clip_pixel_values = new_examples["ref_pixel_values"][-1][0].permute(1, 2, 0)
                new_examples["clip_pixel_values"].append(clip_pixel_values)

            # Limit the number of frames to the same
            new_examples["video_pixel_values"] = torch.stack([example[:batch_video_length] for example in new_examples["video_pixel_values"]])  # B F 3 H W
            new_examples["clip_pixel_values"] = torch.stack([example for example in new_examples["clip_pixel_values"]]) # B H W C
            new_examples["ref_pixel_values"] = torch.stack([example for example in new_examples["ref_pixel_values"]])   # B 1 3 H W
            new_examples["audio_values"] = torch.stack([example[:batch_video_length] for example in new_examples["audio_values"]])  # B F 10 5 384
            
            if new_examples["mask"] is not None:
                new_examples["mask"] = torch.stack(new_examples["mask"])    # B 1 H W

            if new_examples["motion_frame_pixel_values"] is not None:
                new_examples["motion_frame_pixel_values"] = torch.stack(new_examples["motion_frame_pixel_values"])  # B, M, 3, H, W

            return new_examples
        
        except Exception as e:
            print(e)
            for example in examples:
                print(example['video_path'], np.array(example["video_pixel_values"]).shape)
            return None
