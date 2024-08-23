import random
from typing import Sequence, List

from albumentations import DualTransform
from albumentations import KeypointType


class KeypointsTransform(DualTransform):
    def __init__(self, **kwargs):
        super(KeypointsTransform, self).__init__(**kwargs)

    def apply(self, img, **params):
        return img

    def apply_to_bbox(self, bbox, **params):
        raise NotImplementedError("Method apply_to_bbox is not implemented in class " + self.__class__.__name__)

    def apply_to_keypoint(self, keypoint, **params):
        return keypoint

    def apply_to_mask(self, masks, **params):
        return masks


class DropRandomKeypoints(KeypointsTransform):
    def __init__(self, ratio=0.2, **kwargs):
        super(KeypointsTransform, self).__init__(**kwargs)
        self.ratio = ratio

    def apply(self, img, **params):
        return img

    def apply_to_bbox(self, bbox, **params):
        raise NotImplementedError("Method apply_to_bbox is not implemented in class " + self.__class__.__name__)

    def apply_to_keypoints(self, keypoints: Sequence[KeypointType], **params) -> List[KeypointType]:
        return [  # type: ignore
            tuple(self.apply_to_keypoint(list(keypoint), **params))  # type: ignore
            for keypoint in keypoints
        ]

    def apply_to_keypoint(self, keypoint, **params):
        if random.random() <= self.ratio:
            keypoint[-2] = 0.0
        return keypoint

    def apply_to_mask(self, masks, **params):
        return masks


class DropAllKeypoints(DropRandomKeypoints):
    def __init__(self, **kwargs):
        super(DropAllKeypoints, self).__init__(ratio=1, **kwargs)
