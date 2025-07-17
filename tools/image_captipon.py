import os
import torch
import cv2
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

def get_files(roots, suffix=['.png']):
    result = []
    for root in roots:
        for dir, _, files in os.walk(root):
            for file in files:
                if os.path.splitext(file)[1].lower() in suffix:
                    filepath = os.path.join(dir, file)
                    result.append(filepath)
                    if len(result) % 10000 == 0:
                        print(f"num of files: {len(result)}")
    print(f"num of files: {len(result)}")
    assert len(result) > 0
    result.sort()
    return result



class VideoCaption:
    def __init__(self,
                 device = "cuda",
                #  model_path = "/mnt/data/public_ckpt/Qwen2.5-VL-7B-Instruct-AWQ",
                model_path = "/mnt/data/public_ckpt/Qwen2.5-VL-7B-Instruct",
                 ):
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path, torch_dtype="auto", device_map=device)
        self.processor = AutoProcessor.from_pretrained(model_path)

    def __call__(self, image):
        if isinstance(image, str):
            # image = cv2.imread(image)
            image = Image.open(image)

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image,
                    },
                    {"type": "text", "text": "Describe this image."},
                ],
            }
        ]

        # Preparation for inference
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.model.device)

        # Inference: Generation of the output
        generated_ids = self.model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        # print(output_text)
        return output_text[0]


vc = VideoCaption()
root = '/mnt/data/ssd/public_data/omnihuman/test/'

data = []
image_list = get_files([root])
image_list = [
    '/mnt/data/ssd/public_data/omnihuman/test/sing/21.png',
    '/mnt/data/ssd/public_data/omnihuman/test/sing/22.png',
]
for image_path in image_list:
    text_prompt = vc(image_path)

    print(image_path)
    print(text_prompt)

#     item = {
#         "image_path": image_path.replace(root, ''),
#         "audio_path": image_path.replace(root, '').replace('.png', '.wav'),
#         "text_long_prompt": text_prompt,
#         "text_simply_prompt": text_prompt.split('.')[0] + '.',
#     }
#     data.append(item)

# import json
# with open("test.json", "w", encoding="utf-8") as f:
#     json.dump(data, f, ensure_ascii=False, indent=4)

