from __future__ import division, print_function, absolute_import

import cv2
import torch
import numpy as np
from albumentations import (
    Compose, Normalize, ColorJitter, HorizontalFlip, CoarseDropout, RandomCrop, PadIfNeeded, KeypointParams
)
from albumentations.pytorch import ToTensorV2

from torchreid.data.data_augmentation import BIPO
from torchreid.data.data_augmentation.resize import ResizeLongestSide
from torchreid.data.masks_transforms import masks_preprocess_all, AddBackgroundMask, ResizeMasks, PermuteMasksDim, \
    RemoveBackgroundMask
from torchreid.data.masks_transforms.keypoints_transform import DropRandomKeypoints, DropAllKeypoints
from torchreid.data.masks_transforms.resize import Resize


class NpToTensor(object):
    def __call__(self, masks):
        assert isinstance(masks, np.ndarray)
        return torch.as_tensor(masks)

    def __repr__(self):
        return self.__class__.__name__ + '()'


def build_transforms(
    height,
    width,
    config,
    transforms='random_flip',
    norm_mean=[0.485, 0.456, 0.406],
    norm_std=[0.229, 0.224, 0.225],
    remove_background_mask=False,
    masks_preprocess = 'none',
    softmax_weight = 0,
    mask_filtering_threshold = 0.3,
    background_computation_strategy = 'threshold',
    train_dir="",
    **kwargs
):
    """Builds train and test transform functions.

    Args:
        height (int): target image height.
        width (int): target image width.
        transforms (str or list of str, optional): transformations applied to model training.
            Default is 'random_flip'.
        norm_mean (list or None, optional): normalization mean values. Default is ImageNet means.
        norm_std (list or None, optional): normalization standard deviation values. Default is
            ImageNet standard deviation values.
    """
    if transforms is None:
        transforms = []

    if isinstance(transforms, str):
        transforms = [transforms]

    if not isinstance(transforms, list):
        raise ValueError(
            'transforms must be a list of strings, but found to be {}'.format(
                type(transforms)
            )
        )

    if len(transforms) > 0:
        transforms = [t.lower() for t in transforms]

    if norm_mean is None or norm_std is None:
        norm_mean = [0.485, 0.456, 0.406] # imagenet mean
        norm_std = [0.229, 0.224, 0.225] # imagenet std
    normalize = Normalize(mean=norm_mean, std=norm_std, max_pixel_value=255.0)

    print('Building train transforms ...')
    transform_tr = []

    if 'resize_longest_side' in transforms or 'rl' in transforms:
        print('+ resize longest side to {}'.format(height))
        transform_tr += [ResizeLongestSide(height)]
    else:
        print('+ resize to {}x{}'.format(height, width))
        transform_tr += [Resize(height, width, interpolation=config.data.resize.interpolation)]

    if 'bipo' in transforms:
        print('+ BIPO (random occlusion)')
        transform_tr += [BIPO(path=config.data.bipo.path,
                              im_shape=[config.data.height, config.data.width],
                              p=config.data.bipo.p,
                              n=config.data.bipo.n,
                              min_overlap=config.data.bipo.min_overlap,
                              max_overlap=config.data.bipo.max_overlap,
                              pid_sampling_from_batch=config.data.bipo.pid_sampling_from_batch,
                              )]

    if 'random_flip' in transforms or 'rf' in transforms:
        print('+ random flip')
        transform_tr += [HorizontalFlip()]

    if 'random_crop' in transforms or 'rc' in transforms:
        print('+ random crop')
        pad_size = 10
        transform_tr += [PadIfNeeded(min_height=height+pad_size*2, min_width=width+pad_size*2, border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0, p=1),
                         RandomCrop(height, width, p=1)]

    if 'color_jitter' in transforms or 'cj' in transforms:
        print('+ color jitter')
        transform_tr += [
            ColorJitter(brightness=config.data.cj.brightness,
                        contrast=config.data.cj.contrast,
                        saturation=config.data.cj.saturation,
                        hue=config.data.cj.hue,
                        always_apply=config.data.cj.always_apply,
                        p=config.data.cj.p,
                        )
        ]

    print('+ normalization (mean={}, std={})'.format(norm_mean, norm_std))
    transform_tr += [normalize]

    if 'random_erase' in transforms or 're' in transforms:
        print('+ random erase')
        transform_tr += [CoarseDropout(min_holes=1, max_holes=1,  # FIXME: is removing keypoints, should set them invisible
                                       min_height=int(height*0.15), max_height=int(height*0.65),
                                       min_width=int(width*0.15), max_width=int(width*0.65),
                                       fill_value=norm_mean, mask_fill_value=0, always_apply=False, p=0.5)]

    if 'pad_shortest_edge' in transforms or 'ps' in transforms:
        transform_tr += [PadIfNeeded(min_height=height, min_width=height)]

    print('+ to torch tensor of range [0, 1]')
    transform_tr += [ToTensorV2()]

    print('Building test transforms ...')
    print('+ resize to {}x{}'.format(height, width))
    print('+ to torch tensor of range [0, 1]')
    print('+ normalization (mean={}, std={})'.format(norm_mean, norm_std))

    transform_te = [
        Resize(height, width),
        normalize
    ]

    if 'bipo_test' in transforms or 'bipot' in transforms:
        print('+ BIPO test (random occlusion)')
        transform_te += [BIPO(path=config.data.bipo.path,
                              im_shape=[config.data.height, config.data.width],
                              p=config.data.bipo.p,
                              n=config.data.bipo.n,
                              min_overlap=config.data.bipo.min_overlap,
                              max_overlap=config.data.bipo.max_overlap,
                              pid_sampling_from_batch=config.data.bipo.pid_sampling_from_batch,
                              )]

    transform_te += [ToTensorV2()]

    compose_kwargs = {}
    kp_target_transform, kp_prompt_transform = None, None
    if config.model.kpr.keypoints.enabled:
        if 'drop_random_keypoints' in transforms or 'drk' in transforms:
            transform_tr += [DropRandomKeypoints(p=config.data.drk.p, ratio=config.data.drk.ratio)]
        if 'drop_random_keypoints_test' in transforms or 'drkt' in transforms:  # TODO replace by config disable mask prompting
            transform_te += [DropRandomKeypoints(p=config.data.drk.p, ratio=config.data.drk.ratio)]
        if 'drop_all_keypoints' in transforms or 'dak' in transforms:
            transform_tr += [DropAllKeypoints(p=config.data.dak.p)]

        kp_target_transform = [masks_preprocess_all[config.model.kpr.keypoints.target_preprocess]()]
        kp_prompt_transform = [masks_preprocess_all[config.model.kpr.keypoints.prompt_preprocess]()]

        keypoint_params = KeypointParams(format='xy',
                                         label_fields=['kp_vis_score', 'kp_indices'],
                                         remove_invisible=True,
                                         angle_in_degrees=True,
                                         check_each_transform=True)

        compose_kwargs["keypoint_params"] = keypoint_params

        print('+ use add background mask')
        kp_target_transform += [AddBackgroundMask('threshold', -1, 0.05)]
        if config.model.promptable_trans.pose_encoding_strategy == "embed_heatmaps_patches":
            # just cat a background mask that is 1 minus the sum of the keypoints heatmaps.
            prompt_bckg_strategy = 'sum'
            # Keep the other keypoints heatmaps as they are (i.e. no channel wise softmax applied):
            prompt_softmax_weight = -1
            prompt_mask_filtering_threshold = 0.2  # useless here
        else:
            prompt_bckg_strategy = 'threshold'
            prompt_softmax_weight = -1
            prompt_mask_filtering_threshold = 0.2
        kp_prompt_transform += [AddBackgroundMask(prompt_bckg_strategy, prompt_softmax_weight, prompt_mask_filtering_threshold)]

    if config.model.kpr.masks.enabled:
        transform_tr += [PermuteMasksDim()]
        transform_te += [PermuteMasksDim()]

        if remove_background_mask:  # ISP masks
            print('+ use remove background mask')
            # remove background before performing other transforms
            transform_tr = [RemoveBackgroundMask()] + transform_tr
            transform_te = [RemoveBackgroundMask()] + transform_te

            # Derive background mask from all foreground masks once other tasks have been performed
            print('+ use add background mask')
            transform_tr += [AddBackgroundMask('sum')]
            transform_te += [AddBackgroundMask('sum')]
        else:  # Pifpaf confidence based masks
            if masks_preprocess != 'none':
                print('+ masks preprocess = {}'.format(masks_preprocess))
                masks_preprocess_transform = masks_preprocess_all[masks_preprocess]
                # mask grouping as first transform to reduce tensor size asap and speed up other transforms
                transform_tr = [masks_preprocess_transform()] + transform_tr
                transform_te = [masks_preprocess_transform()] + transform_te

            print('+ use add background mask')
            transform_tr += [AddBackgroundMask(background_computation_strategy, softmax_weight, mask_filtering_threshold)]
            transform_te += [AddBackgroundMask(background_computation_strategy, softmax_weight, mask_filtering_threshold)]

        transform_tr += [ResizeMasks(config.data.resize.mask_interpolation)]
        transform_te += [ResizeMasks(config.data.resize.mask_interpolation)]

    transform_tr = Compose(transform_tr, is_check_shapes=False, **compose_kwargs)
    transform_te = Compose(transform_te, is_check_shapes=False, **compose_kwargs)

    return transform_tr, transform_te, kp_target_transform, kp_prompt_transform
