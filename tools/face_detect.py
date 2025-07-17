import cv2
import numpy as np
from PIL import Image
import torch
from torchvision import transforms

from .scrfd import SCRFD


class FaceDetector:
    def __init__(self, 
                 gpu_id=0,
                 det_path="/mnt/data/public_ckpt/face_detect/scrfd_10g_shape640x640.onnx"):
        self.detector = SCRFD(model_path=det_path, gpu_id=gpu_id)

    def convert_inp(self, inp):
        if isinstance(inp, str):
            img = cv2.imread(inp)
        elif isinstance(inp, np.ndarray):
            img = inp.copy()
        elif isinstance(inp, Image.Image):
            img = np.array(inp)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        else:
            raise TypeError("xxx")
        return img

    def get_face_box(self, img):
        h, w = img.shape[:2]
        boxes, _ = self.detector.detect(img)
        if len(boxes) == 0:
            return boxes
        
        boxes = boxes[:, :4].astype(np.int32)
        boxes[:, 0] = np.clip(boxes[:, 0], 0, w - 1)    # x1
        boxes[:, 1] = np.clip(boxes[:, 1], 0, h - 1)    # y1
        boxes[:, 2] = np.clip(boxes[:, 2], 0, w - 1)    # x2
        boxes[:, 3] = np.clip(boxes[:, 3], 0, h - 1)    # y2
        return boxes
    
    def expand_box(self, bbox, ratio, height, width):
        ratio_x1, ratio_y1, ratio_x2, ratio_y2 = ratio

        bbox_h = bbox[3] - bbox[1]
        bbox_w = bbox[2] - bbox[0]
        
        expand_x1 = max(bbox[0] - ratio_x1 * bbox_w, 0)
        expand_y1 = max(bbox[1] - ratio_y1 * bbox_h, 0)
        expand_x2 = min(bbox[2] + ratio_x2 * bbox_w, width)
        expand_y2 = min(bbox[3] + ratio_y2 * 2 * bbox_h, height)

        return [expand_x1,expand_y1,expand_x2,expand_y2]
    
    def expand_to_square(self, bbox, height=None, width=None):
        # 计算原始边界框的高度和宽度
        h = bbox[3] - bbox[1]
        w = bbox[2] - bbox[0]

        # 找到中心点坐标
        c_h = (bbox[1] + bbox[3]) / 2
        c_w = (bbox[0] + bbox[2]) / 2

        # 确定正方形的边长为原边界框的最大维度
        side = max(h, w)

        # 计算新的边界框坐标
        square_x1 = c_w - side / 2
        square_y1 = c_h - side / 2
        square_x2 = c_w + side / 2
        square_y2 = c_h + side / 2

        # 如果提供了图像的高度和宽度，则确保边界框不会超出图像范围
        if height is not None and width is not None:
            square_x1 = max(0, min(square_x1, width))
            square_y1 = max(0, min(square_y1, height))
            square_x2 = max(0, min(square_x2, width))
            square_y2 = max(0, min(square_y2, height))

        return [int(square_x1), int(square_y1), int(square_x2), int(square_y2)]
    
    def sonic_face_crop(self, inp, return_type='PIL'):
        # 线上sonic的图片裁剪逻辑
        img = self.convert_inp(inp)
        imgh, imgw = img.shape[:2]

        boxes = self.get_face_box(img)
        if len(boxes) == 0:
            return None
        
        # 单人处理
        bbox = boxes[0]
        bbox = self.expand_box(bbox, ratio=[0.5, 0.5, 0.5, 0.5], height=imgh, width=imgw)
        bbox = self.expand_to_square(bbox, height=imgh, width=imgw)
        
        crop = img[bbox[1]:bbox[3], bbox[0]:bbox[2]]
        if return_type == 'PIL':
            crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            crop = Image.fromarray(crop)
        return crop

    def get_face_mask(self, inp, expand_ratio=1.1, multi_face=False):
        img = self.convert_inp(inp)
        imgh, imgw = img.shape[:2]

        boxes = self.get_face_box(img)

        if len(boxes) == 0:
            return None

        if not multi_face:
            boxes = [boxes[0]]

        # face mask
        mask = np.zeros((imgh, imgw), dtype=np.uint8)

        for box in boxes:
            x1, y1, x2, y2 = box
            ww = x2 - x1
            hh = y2 - y1
            ww, hh = (x2-x1) * expand_ratio, (y2-y1) * expand_ratio

            center = [(x2+x1)//2, (y2+y1)//2]
            x1 = max(center[0] - ww//2, 0)
            y1 = max(center[1] - hh//2, 0)
            x2 = min(center[0] + ww//2, imgw)
            y2 = min(center[1] + hh//2, imgh)
            mask[int(y1):int(y2), int(x1):int(x2)] = 255

        mask = mask[None]
        mask = torch.from_numpy(mask)
        mask = mask / 255
        mask = mask.unsqueeze(0)

        return mask
