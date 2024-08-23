import numpy as np
import cv2
import torch
from albumentations import (
    DualTransform
)
from segment_anything.utils.transforms import ResizeLongestSide as ResizeLongestSideSAM
from albumentations.augmentations.geometric import functional as F


class ResizeLongestSide(DualTransform):
    """
    Resizes images to longest side 'target_length', as well as provides
    methods for resizing coordinates and boxes.
    Delegate to SAM ResizeLongestSide implementation.
    """

    def __init__(self, target_length: int, **kwargs) -> None:
        super(DualTransform, self).__init__(**kwargs)
        self.target_length = target_length
        self.transform = ResizeLongestSideSAM(target_length)

    def apply(self, img, interpolation=cv2.INTER_LINEAR, **params):
        if torch.is_tensor(img):
            return self.transform.apply_image_torch(img)
        else:
            return self.transform.apply_image(img)

    def apply_to_bbox(self, bbox, **params):
        original_size = params["original_size"]
        if torch.is_tensor(bbox):
            return self.transform.apply_boxes_torch(bbox, original_size)
        else:
            return self.transform.apply_boxes(bbox, original_size)

    def apply_to_keypoint(self, keypoint, **params):  # FIXME keypoint or keypoints?
        original_size = params["original_size"]
        if torch.is_tensor(keypoint):
            return self.transform.apply_coords_torch(keypoint, original_size)
        else:
            return self.transform.apply_coords(keypoint, original_size)

    def get_transform_init_args_names(self):
        return ("target_length")

    def apply_to_mask(self, img: np.ndarray, **params) -> np.ndarray:
        # from albumentations/augmentations/geometric/resize/LongestMaxSize :
        return F.longest_max_size(img, max_size=self.target_length, interpolation=cv2.INTER_LINEAR)
