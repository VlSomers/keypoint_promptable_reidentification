from __future__ import print_function, absolute_import

from .datasets import (
    Dataset, ImageDataset, VideoDataset, register_image_dataset,
    register_video_dataset, get_dataset_nickname, get_image_dataset
)
from .datamanager import ImageDataManager, VideoDataManager
from .datasets.keypoints_to_masks import parts_info_per_strat
from .masks_transforms import masks_preprocess_all


def compute_parts_num_and_names(cfg):
    mask_config = get_image_dataset(cfg.data.sources[0]).get_masks_config(
        cfg.model.kpr.masks.dir if not (cfg.model.kpr.keypoints.enabled and cfg.model.kpr.keypoints.target_masks != 'none') else cfg.model.kpr.keypoints.target_masks
    )
    if cfg.loss.name == "part_based":
        if cfg.model.kpr.keypoints.enabled:
            kp_cfg = cfg.model.kpr.keypoints
            masks_cfg = cfg.model.kpr.masks
            if kp_cfg.target_masks != 'none':
                masks_cfg.parts_num, masks_cfg.parts_names = get_parts_num_and_names(kp_cfg.target_preprocess, kp_cfg.target_masks)
            if kp_cfg.prompt_masks != 'none':
                masks_cfg.prompt_parts_num, masks_cfg.prompt_parts_names = get_parts_num_and_names(kp_cfg.prompt_preprocess, kp_cfg.prompt_masks)
        if cfg.model.kpr.masks.enabled:
            masks_transform = cfg.model.kpr.masks.preprocess
            if (
                mask_config is not None and mask_config[1]
            ) or masks_transform == "none":
                # ISP masks or no transform
                cfg.model.kpr.masks.parts_num = mask_config[0]
                cfg.model.kpr.masks.parts_names = (
                    mask_config[3]
                    if 3 in mask_config
                    else [
                        "p{}".format(p)
                        for p in range(1, cfg.model.kpr.masks.parts_num + 1)
                    ]
                )
                if not cfg.model.kpr.keypoints.enabled or \
                        (cfg.model.kpr.keypoints.enabled and cfg.model.kpr.keypoints.prompt_masks == 'none'):
                    cfg.model.kpr.masks.prompt_parts_num = cfg.model.kpr.masks.parts_num
                    cfg.model.kpr.masks.prompt_parts_names = cfg.model.kpr.masks.parts_names
            else:
                if not cfg.model.kpr.keypoints.enabled or \
                        (cfg.model.kpr.keypoints.enabled and cfg.model.kpr.keypoints.target_masks == 'none'):
                    masks_transform = masks_preprocess_all[masks_transform]()
                    cfg.model.kpr.masks.parts_num = masks_transform.parts_num
                    cfg.model.kpr.masks.parts_names = masks_transform.parts_names
            if cfg.model.promptable_trans.masks_prompting:
                if not cfg.model.kpr.keypoints.enabled or \
                        (cfg.model.kpr.keypoints.enabled and cfg.model.kpr.keypoints.prompt_masks == 'none'):
                    # cfg.model.kpr.masks.prompt_parts_num = cfg.model.kpr.masks.parts_num
                    cfg.model.kpr.masks.prompt_parts_num = 1  # FIXME only supports segmentation masks now, no multi-channel parsing labels
                    cfg.model.kpr.masks.prompt_parts_names = cfg.model.kpr.masks.parts_names


def get_parts_num_and_names(masks_transform, masks_strat):
    if masks_transform == "none":
        parts_num = parts_info_per_strat[masks_strat][0]
        parts_names = list(parts_info_per_strat[masks_strat][1])
    else:
        parts_num = masks_preprocess_all[masks_transform]().parts_num
        parts_names = list(masks_preprocess_all[masks_transform]().parts_grouping.keys())
    return parts_num, parts_names
