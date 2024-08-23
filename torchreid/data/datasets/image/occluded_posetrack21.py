from __future__ import absolute_import, division, print_function

import json
import os
from typing import Any
import re
import cv2
import pandas as pd
import torch.nn.functional as F
from dataclasses import dataclass
from torch.utils.data import Dataset
from math import ceil
from pathlib import Path
from skimage.transform import resize
from tqdm import tqdm
from yacs.config import CfgNode as CN
import sys
from torch.utils.data.dataloader import default_collate, DataLoader
from abc import abstractmethod
import torch
import numpy as np
from PIL import Image
from omegaconf.listconfig import ListConfig
import logging
import os.path as osp
from abc import ABC, ABCMeta
from segment_anything import SamPredictor, sam_model_registry
from ..dataset import ImageDataset
from ...datasets.keypoints_to_masks import kp_img_to_kp_bbox, rescale_keypoints
from ....data.masks_transforms import CocoToEightBodyMasks
from ....utils.imagetools import build_keypoints_heatmaps, build_keypoints_gaussian_heatmaps, \
    build_joints_heatmaps, build_joints_gaussian_heatmaps, gkern
from ....utils.visualization.visualize_query_gallery_rankings import colored_body_parts_overlay, draw_keypoints

log = logging.getLogger(__name__)

# This code os borrowed from Tracklab: https://github.com/TrackingLaboratory/tracklab
# The original purpose of this Tracklab code is to build a ReID dataset from a MOT dataset
# We just copy pasted the relevant parts from Tracklab and adapted them to turn the PoseTrack21 dataset into the
# Occluded-PoseTrack-ReID dataset. This code will generate an new 'reid' folder inside the PoseTrack21 dataset folder,
# containing the ReID dataset, i.e. persons crops, keypoints, and masks.
# This class employs the ground keypoints from PoseTrack21 as prompts, and PifPaf and SAM to generate the pseudo
# human-parsing labels.

class OccludedPosetrack21(ImageDataset):
    img_ext = ".jpg"
    masks_ext = ".npy"
    reid_dir = "occluded_posetrack_reid"
    reid_images_dir = "images"
    reid_masks_dir = "masks"
    reid_fig_dir = "figures"
    reid_anns_dir = "anns"
    images_anns_filename = "reid_crops_anns.json"
    masks_anns_filename = "reid_masks_anns.json"
    dataset_sampling_filename = "dataset_sampling.json"
    train_dir = 'gaussian_joints'
    dataset_dir = "PoseTrack21"

    masks_dirs = {
        # dir_name: (masks_stack_size, contains_background_mask)
        "keypoints": (17, False, ".npy", ["p{}".format(p) for p in range(1, 17)],),
        "keypoints_gaussian": (17, False, ".npy", ["p{}".format(p) for p in range(1, 17)],),
        "joints": (10, False, ".npy", ["p{}".format(p) for p in range(1, 17)]),
        "joints_gaussian": (10, False, ".npy", ["p{}".format(p) for p in range(1, 17)]),
        "pose_on_img": (35, False, ".npy", ["p{}".format(p) for p in range(1, 35)]),
        "pose_on_img_crops": (35, False, ".npy", ["p{}".format(p) for p in range(1, 35)]),
    }

    reid_config = CN()
    reid_config.name = "PoseTrack21"
    reid_config.nickname = "pt21"
    reid_config.fig_size = [384, 128]
    reid_config.mask_size = [96, 32]
    reid_config.max_crop_size = [384, 128]
    reid_config.masks_mode = "pose_on_img_crops"  # "keypoints", "keypoints_gaussian", "joints", "joints_gaussian"
    reid_config.eval_metric = "mot_inter_intra_video"  # {"mot_inter_intra_video", "mot_intra_video", "mot_inter_video"}
    reid_config.multi_video_queries_only = False  # will be set to True by default if eval_metric is "mot_inter_video"
    reid_config.enable_human_parsing_labels = True
    reid_config.columns = []
    reid_config.train = CN()
    reid_config.train.set_name = "train"
    reid_config.train.min_vis = 0.3
    reid_config.train.min_h = 10
    reid_config.train.min_w = 10
    reid_config.train.min_samples_per_id = 4
    reid_config.train.max_samples_per_id = 20
    reid_config.train.max_total_ids = 1000
    reid_config.test = CN()
    reid_config.test.set_name = "val"
    reid_config.test.min_vis= 0.
    reid_config.test.min_h= 0
    reid_config.test.min_w= 0
    reid_config.test.min_samples_per_id= 4
    reid_config.test.max_samples_per_id= 10
    reid_config.test.max_total_ids= -1
    reid_config.test.ratio_query_per_id= 0.2

    pifpaf_config = CN()
    pifpaf_config.predict = CN()
    pifpaf_config.predict["checkpoint"] = "shufflenetv2k30"
    pifpaf_config.predict["long-edge"] = 256
    pifpaf_config.predict["quiet"] = None
    pifpaf_config.predict["dense-connections"] = None
    pifpaf_config.predict["seed-threshold"] = 0.2
    pifpaf_config.predict["instance-threshold"] = 0.15
    pifpaf_config.predict["decoder-workers"] = 8

    @staticmethod
    def get_masks_config(masks_dir):
        if masks_dir not in OccludedPosetrack21.masks_dirs:
            return None
        else:
            return OccludedPosetrack21.masks_dirs[masks_dir]

    def gallery_filter(self, q_pid, q_camid, q_ann, g_pids, g_camids, g_anns):
        """camid refers to video id: remove gallery samples from the different videos than query sample"""
        if self.eval_metric == 'mot_inter_intra_video':
            return np.array(np.zeros_like(g_pids), dtype=bool)
        elif self.eval_metric == 'mot_inter_video':
            remove = g_camids == q_camid
            return remove
        elif self.eval_metric == 'mot_intra_video':
            remove = g_camids != q_camid
            return remove
        else:
            raise ValueError

    def __init__(
        self,
        masks_dir="",
        root="",
        occluded_dataset=True,  # sample most occluded images as queries in the test set
        config=None,
        **kwargs
    ):
        self.root = osp.abspath(osp.expanduser(root))
        self.dataset_dir = osp.join(self.root, self.dataset_dir)
        # Init
        self.tracking_dataset = PoseTrack21(
            dataset_path=self.dataset_dir,
            annotation_path=Path(self.dataset_dir, "posetrack_data"),
        )
        self.pose_model = None
        # self.pose_model = OpenPifPaf(self.pifpaf_config,  # required to generate dataset and annotations in the first place
        #                              device="cuda" if torch.cuda.is_available() else "cpu")

        self.pose_dl = None
        self.pose_datapipe = None
        self.dataset_path = Path(self.tracking_dataset.dataset_path)
        self.masks_dir = masks_dir
        self.column_mapping = {}

        self.eval_metric = self.reid_config.eval_metric
        self.multi_video_queries_only = self.reid_config.multi_video_queries_only

        val_set = self.tracking_dataset.sets[self.reid_config.test.set_name]
        train_set = self.tracking_dataset.sets[self.reid_config.train.set_name]

        self.occluded_dataset = occluded_dataset
        self.occ_pt_config = config.occluded_posetrack
        self.sam_checkpoint = osp.abspath(osp.expanduser(self.occ_pt_config.sam_checkpoint))
        self.enable_sam = self.occ_pt_config.enable_sam
        self.enable_dataset_sampling_loading = self.occ_pt_config.enable_dataset_sampling_loading

        assert (
            self.reid_config.train.max_samples_per_id
            >= self.reid_config.train.min_samples_per_id
        ), "max_samples_per_id must be >= min_samples_per_id"
        assert (
            self.reid_config.test.max_samples_per_id
            >= self.reid_config.test.min_samples_per_id
        ), "max_samples_per_id must be >= min_samples_per_id"

        if self.masks_dir in self.masks_dirs:
            (
                self.masks_parts_numbers,
                self.has_background,
                self.masks_suffix,
                self.masks_parts_names,
            ) = self.masks_dirs[self.masks_dir]
        else:
            (
                self.masks_parts_numbers,
                self.has_background,
                self.masks_suffix,
                self.masks_parts_names,
            ) = (None, None, None, None)

        # Build ReID dataset from MOT dataset
        self.build_reid_set(
            train_set,
            self.reid_config,
            "train",
            is_test_set=False,
        )

        self.build_reid_set(
            val_set,
            self.reid_config,
            "val",
            is_test_set=True,
        )

        self.train_gt_dets = train_set.detections_gt
        self.val_gt_dets = val_set.detections_gt

        # Get train/query/gallery sets as torchreid list format
        self.train_df = self.train_gt_dets[self.train_gt_dets["split"] == "train"]
        self.query_df = self.val_gt_dets[self.val_gt_dets["split"] == "query"]
        self.gallery_df = self.val_gt_dets[self.val_gt_dets["split"] == "gallery"]
        assert len(self.train_df) > 0, "An error occurred, no train samples found"
        assert len(self.query_df) > 0, "An error occurred, no query samples found"
        assert len(self.gallery_df) > 0, "An error occurred, no gallery samples found"

        train, query, gallery = self.to_torchreid_dataset_format(
            [self.train_df, self.query_df, self.gallery_df]
        )

        super().__init__(train, query, gallery, config=config, **kwargs)

    def build_reid_set(self, tracking_set, reid_config, split, is_test_set):
        """
        Build ReID metadata for a given MOT dataset split.
        Only a subset of all MOT groundtruth detections is used for ReID.
        Detections to be used for ReID are selected according to the filtering criteria specified in the config 'reid_cfg'.
        If "enable_dataset_sampling_loading" is set, the sampling annotations are loaded from disk to assign each
        detection a "split" value, that can be "train"/"none" for the train set and "query"/"gallery"/"none" for the test
         set (ReID test set = tracking validation set).
        Image crops and human parsing labels (masks) are generated for each selected detection only.
        If the config is changed and more detections are selected, the image crops and masks are generated only for
        these new detections.
        """
        image_metadatas = tracking_set.image_metadatas
        detections = tracking_set.detections_gt
        fig_size = reid_config.fig_size
        mask_size = reid_config.mask_size
        max_crop_size = reid_config.max_crop_size
        reid_set_cfg = reid_config.test if is_test_set else reid_config.train
        masks_mode = reid_config.masks_mode

        log.info("Loading {} set...".format(split))

        # Precompute all paths
        reid_path = Path(self.dataset_path, self.reid_dir) if self.reid_config.enable_human_parsing_labels else Path(self.dataset_path, self.reid_dir)
        reid_img_path = reid_path / self.reid_images_dir / split
        reid_mask_path = reid_path / self.reid_masks_dir / split
        reid_fig_path = reid_path / self.reid_fig_dir / split
        reid_anns_filepath = (
            reid_path
            / self.reid_images_dir
            / self.reid_anns_dir
            / (split + "_" + self.images_anns_filename)
        )
        masks_anns_filepath = (
            reid_path
            / self.reid_masks_dir
            / self.reid_anns_dir
            / (split + "_" + self.masks_anns_filename)
        )
        dataset_sampling_path = Path(self.dataset_path, self.reid_dir) / (split + "_" + self.dataset_sampling_filename)

        # Load reid crops metadata into existing ground truth detections dataframe
        self.load_reid_annotations(
            detections,
            reid_anns_filepath,
            ["reid_crop_path", "reid_crop_width", "reid_crop_height", "negative_kps"],
        )

        # Add negative keypoints to each detection
        detections["negative_kps"] = detections["negative_kps"].apply(lambda x: np.array(x) if (isinstance(x, list) and len(x) > 0) else np.empty((0, 17, 3)))

        # Load reid masks metadata into existing ground truth detections dataframe
        self.load_reid_annotations(detections, masks_anns_filepath, ["masks_path"])
        #
        # masks_anns_filepath = (
        #     reid_path
        #     / self.reid_masks_dir
        #     / self.reid_anns_dir
        #     / (split + "____" + self.masks_anns_filename)
        # )
        #
        # log.info(
        #     '################################## Saving reid human parsing annotations as json to "{}"'.format(
        #         masks_anns_filepath
        #     )
        # )
        # print(f"################################## Saving {masks_anns_filepath}")
        # print(f"################################## mask path {reid_mask_path}")
        # masks_anns_filepath.parent.mkdir(parents=True, exist_ok=True)
        # # remove prefix inside var "reid_mask_path" from masks_path:
        # detections["masks_path"] = detections["masks_path"].apply(lambda x: x.replace(str(reid_mask_path) + "/pose_on_img_crops/s", "") if x else None)
        # print(detections["masks_path"][detections["masks_path"].notnull()])
        # detections[["id", "masks_path"]].to_json(masks_anns_filepath)
        #
        # return
        # Sampling of detections to be used to create the ReID dataset
        if self.enable_dataset_sampling_loading:
            self.load_dataset_sampling(detections, dataset_sampling_path)
        else:
            self.sample_detections_for_reid(detections, reid_set_cfg)

        # Save ReID detections crops and related metadata. Apply only on sampled detections
        self.save_reid_img_crops(
            detections,
            reid_img_path,
            split,
            reid_anns_filepath,
            image_metadatas,
            max_crop_size,
        )

        # Save human parsing pseudo ground truth and related metadata. Apply only on sampled detections
        if self.reid_config.enable_human_parsing_labels:
            self.save_reid_masks_crops(
                detections,
                reid_img_path,
                reid_mask_path,
                reid_fig_path,
                split,
                masks_anns_filepath,
                image_metadatas,
                fig_size,
                mask_size,
                mode=masks_mode,
            )
        else:
            detections["masks_path"] = ''

        # Add 0-based pid column (for Torchreid compatibility) to sampled detections
        self.add_pid_column(detections)
        self.add_occlusion_level_column(detections)

        # Flag sampled detection as a query or gallery if this is a test set
        if is_test_set:
            self.query_gallery_split(detections, reid_set_cfg.ratio_query_per_id)

        # Save selected detections metadata to disk
        # self.save_dataset_sampling(detections, dataset_sampling_path)

        # Turn path into absolute path
        detections['masks_path'] = detections['masks_path'].apply(lambda x: str(reid_mask_path / x) if x else None)
        detections['reid_crop_path'] = detections['reid_crop_path'].apply(lambda x: str(reid_img_path / x) if x else None)

    def save_dataset_sampling(self, detections, dataset_sampling_path):
        log.info(
            'Saving dataset sampling annotations as json to "{}"'.format(dataset_sampling_path)
        )
        dataset_sampling_path.parent.mkdir(parents=True, exist_ok=True)
        detections[
            ["id", "split"]
        ].to_json(dataset_sampling_path)

    def add_negative_samples(self, _df):
        all_kps_in_img = np.array(list(_df.keypoints_xyc))
        id_to_index = {k: v for v, k in enumerate(list(_df.id))}
        _df["negative_kps"] = _df\
            .apply(lambda bb: keypoints_in_bbox_coord(np.delete(all_kps_in_img, id_to_index[bb.id], axis=0), bb.bbox_ltwh), axis=1)\
            .apply(lambda kp_xyc_bbox: kp_xyc_bbox[kp_xyc_bbox[:, :, 2].sum(axis=1) > 0]) # remove non visibile skeletons

        return _df

    def load_reid_annotations(self, gt_dets, reid_anns_filepath, columns):
        if reid_anns_filepath.exists():
            reid_anns = pd.read_json(
                reid_anns_filepath, convert_dates=False, convert_axes=False
            )
            reid_anns.set_index("id", drop=False, inplace=True)
            tmp_df = gt_dets.merge(
                reid_anns,
                left_index=True,
                right_index=True,
                validate="one_to_one",
            )
            gt_dets[columns] = tmp_df[columns]
        else:
            # no annotations yet, initialize empty columns
            for col in columns:
                gt_dets[col] = None

    def load_dataset_sampling(self, dets_df, dataset_sampling_path):
        if dataset_sampling_path.exists():
            sampling_anns = pd.read_json(
                dataset_sampling_path, convert_dates=False, convert_axes=False
            )
            sampling_anns.set_index("id", drop=False, inplace=True)

            # Drop the 'split' column since it should be overwritten by the sampling file
            if "split" in dets_df.columns:
                dets_df.drop(columns=['split'], inplace=True)

            tmp_df = dets_df.merge(
                sampling_anns,
                left_index=True,
                right_index=True,
                validate="one_to_one",
            )
            dets_df["split"] = tmp_df["split"]
        else:
            raise FileNotFoundError("Dataset sampling file not found ({}). Please follow the instructions on the main repository to download the file and place it under this location.".format(dataset_sampling_path))

    def sample_detections_for_reid(self, dets_df, reid_cfg):
        dets_df["split"] = "none"

        # Filter detections by visibility
        dets_df_f1 = dets_df[dets_df.visibility >= reid_cfg.min_vis]

        # Filter detections by crop size
        keep = dets_df_f1.bbox_ltwh.apply(
            lambda x: x[2] > reid_cfg.min_w
        ) & dets_df_f1.bbox_ltwh.apply(lambda x: x[3] > reid_cfg.min_h)
        dets_df_f2 = dets_df_f1[keep]
        log.warning(
            "{} removed because too small samples (h<{} or w<{}) = {}".format(
                self.__class__.__name__,
                (reid_cfg.min_h),
                (reid_cfg.min_w),
                len(dets_df_f1) - len(dets_df_f2),
            )
        )

        # Filter detections by uniform sampling along each tracklet
        dets_df_f3 = (
            dets_df_f2.groupby("person_id")
            .apply(
                self.uniform_tracklet_sampling, reid_cfg.max_samples_per_id, "image_id"
            )
            .reset_index(drop=True)
            .copy()
        )
        log.warning(
            "{} removed for uniform tracklet sampling = {}".format(
                self.__class__.__name__, len(dets_df_f2) - len(dets_df_f3)
            )
        )

        # Keep only ids with at least MIN_SAMPLES appearances
        count_per_id = dets_df_f3.person_id.value_counts()
        ids_to_keep = count_per_id.index[count_per_id.ge((reid_cfg.min_samples_per_id))]
        dets_df_f4 = dets_df_f3[dets_df_f3.person_id.isin(ids_to_keep)]
        log.warning(
            "{} removed for not enough samples per id = {}".format(
                self.__class__.__name__, len(dets_df_f3) - len(dets_df_f4)
            )
        )

        # Keep only max_total_ids ids
        if reid_cfg.max_total_ids == -1 or reid_cfg.max_total_ids > len(
            dets_df_f4.person_id.unique()
        ):
            reid_cfg.max_total_ids = len(dets_df_f4.person_id.unique())
        # reset seed to make sure the same split is used if the dataset is instantiated multiple times
        np.random.seed(0)
        ids_to_keep = np.random.choice(
            dets_df_f4.person_id.unique(), replace=False, size=reid_cfg.max_total_ids
        )
        dets_df_f5 = dets_df_f4[dets_df_f4.person_id.isin(ids_to_keep)]

        dets_df.loc[dets_df.id.isin(dets_df_f5.id), "split"] = "train"
        log.info(
            "{} filtered size = {}".format(self.__class__.__name__, len(dets_df_f5))
        )

    def save_reid_img_crops(
        self,
        gt_dets,
        save_path,
        set_name,
        reid_anns_filepath,
        metadatas_df,
        max_crop_size,
    ):
        """
        Save on disk all detections image crops from the ground truth dataset to build the reid dataset.
        Create a json annotation file with crops metadata.
        """
        save_path = save_path
        max_h, max_w = max_crop_size
        gt_dets_for_reid = gt_dets[
            (gt_dets.split != "none") & gt_dets.reid_crop_path.isnull()
        ]
        if len(gt_dets_for_reid) == 0:
            log.info(
                "All detections used for ReID already have their image crop saved on disk."
            )
            return

        # compute negative keypoints to be saved on disk
        gt_dets["negative_kps"] = gt_dets.groupby("image_id").apply(self.add_negative_samples).reset_index(level=0, drop=True)["negative_kps"]
        gt_dets_for_reid = gt_dets[
            (gt_dets.split != "none") & gt_dets.reid_crop_path.isnull()
        ]
        # gt_dets_for_reid.reset_index(drop=True, inplace=True)
        grp_gt_dets = gt_dets_for_reid.groupby(["video_id", "image_id"])
        with tqdm(
            total=len(gt_dets_for_reid),
            desc="Extracting all {} reid crops".format(set_name),
        ) as pbar:
            for (video_id, image_id), dets_from_img in grp_gt_dets:
                img_metadata = metadatas_df[metadatas_df.id == image_id].iloc[0]
                img = cv2.imread(img_metadata.file_path)
                for det_metadata in dets_from_img.itertuples():
                    # crop and resize bbox from image
                    bbox_ltwh = det_metadata.bbox_ltwh
                    bbox_ltwh = clip_bbox_ltwh_to_img_dim(
                        bbox_ltwh, img.shape[1], img.shape[0]
                    )
                    pid = det_metadata.person_id
                    l, t, w, h = bbox_ltwh.astype(int)
                    img_crop = img[t : t + h, l : l + w]
                    if h > max_h or w > max_w:
                        img_crop = cv2.resize(img_crop, (max_w, max_h), cv2.INTER_CUBIC)

                    # save crop to disk
                    filename = "{}_{}_{}{}".format(
                        pid, video_id, img_metadata.id, self.img_ext
                    )
                    rel_filepath = Path(str(video_id), filename)
                    abs_filepath = Path(save_path, rel_filepath)
                    abs_filepath.parent.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(abs_filepath), img_crop)

                    # save image crop metadata
                    gt_dets.at[det_metadata.Index, "reid_crop_path"] = str(rel_filepath)
                    gt_dets.at[det_metadata.Index, "reid_crop_width"] = img_crop.shape[1]
                    gt_dets.at[det_metadata.Index, "reid_crop_height"] = img_crop.shape[0]
                    pbar.update(1)

        log.info(
            'Saving reid crops annotations as json to "{}"'.format(reid_anns_filepath)
        )
        reid_anns_filepath.parent.mkdir(parents=True, exist_ok=True)
        gt_dets[
            ["id", "reid_crop_path", "reid_crop_width", "reid_crop_height", "negative_kps"]
        ].to_json(reid_anns_filepath)

    def save_reid_masks_crops(
        self,
        gt_dets,
        reid_img_path,
        masks_save_path,
        fig_save_path,
        set_name,
        reid_anns_filepath,
        metadatas_df,
        fig_size,
        masks_size,
        mode="keypoints_gaussian",
    ):
        """
        Save on disk all human parsing gt for each reid crop.
        Create a json annotation file with human parsing metadata.
        """
        fig_h, fig_w = fig_size
        mask_h, mask_w = masks_size
        g_scale = 10
        g_radius = int(mask_w / g_scale)
        gaussian = gkern(g_radius * 2 + 1)
        gt_dets_for_reid = gt_dets[
            (gt_dets.split != "none") & gt_dets.masks_path.isnull()
        ]
        if mode == "none":
            log.info("No human parsing labels to compute for this mode.")
            return
        if len(gt_dets_for_reid) == 0:
            log.info("All reid crops already have human parsing masks labels.")
            return
        if (mode == "pose_on_img_crops" or mode == "pose_on_img") and self.enable_sam:
            model_type = "vit_h"
            sam = sam_model_registry[model_type](checkpoint=self.sam_checkpoint)
            sam.to(device="cuda" if torch.cuda.is_available() else "cpu")
            predictor = SamPredictor(sam)
        kp_grouping_eight_bp = CocoToEightBodyMasks()
        # kp_grouping_eight_bp = None
        grp_gt_dets = gt_dets_for_reid.groupby(["video_id", "image_id"])
        with tqdm(
            total=len(gt_dets_for_reid),
            desc="Extracting all {} human parsing labels".format(set_name),
        ) as pbar:
            for (video_id, image_id), dets_from_img in grp_gt_dets:
                img_metadata = metadatas_df[metadatas_df.id == image_id].iloc[0]
                # load image once to get video frame size
                if mode == "pose_on_img":
                    if self.pose_dl == None:  # TODO
                        self.pose_dl = DataLoader(
                            dataset=self.pose_datapipe,
                            batch_size=128,
                            num_workers=0,
                            collate_fn=type(self.pose_model).collate_fn,
                            persistent_workers=False,
                        )
                    fields_list = []
                    self.pose_datapipe.update(
                        metadatas_df[metadatas_df.id == image_id], None
                    )
                    for idxs, pose_batch in self.pose_dl:
                        batch_metadatas = metadatas_df.loc[idxs]
                        _, fields = self.pose_model.process(
                            pose_batch, batch_metadatas, return_fields=True
                        )
                        fields_list.extend(fields)

                    masks_gt_or = torch.concat(
                        (
                            fields_list[0][0][:, 1],
                            fields_list[0][1][:, 1],
                        )
                    )
                    img = cv2.imread(img_metadata.file_path)
                    masks_gt = resize(
                        masks_gt_or.numpy(),
                        (masks_gt_or.numpy().shape[0], img.shape[0], img.shape[1]),
                    )

                # loop on detections in frame
                for det_metadata in dets_from_img.itertuples():
                    img_crop = cv2.imread(str(Path(reid_img_path, det_metadata.reid_crop_path)))
                    img_crop = cv2.resize(img_crop, (fig_w, fig_h), cv2.INTER_CUBIC)
                    l, t, w, h = det_metadata.bbox_ltwh
                    kps_xyc_or = kp_img_to_kp_bbox(det_metadata.keypoints_xyc, det_metadata.bbox_ltwh)
                    keypoints_xyc = rescale_keypoints(
                        kps_xyc_or,
                        (w, h),
                        (mask_w, mask_h),
                    )
                    assert ((keypoints_xyc[:, 0] >= 0) & (keypoints_xyc[:, 0] < mask_w)).all()
                    assert ((keypoints_xyc[:, 1] >= 0) & (keypoints_xyc[:, 1] < mask_h)).all()

                    keypoints_xyc_crop = clip_keypoints_to_image(kps_xyc_or, (w, h))
                    keypoints_xyc_crop = rescale_keypoints(keypoints_xyc_crop, (w, h), (fig_w, fig_h))

                    negative_kps_xyc = det_metadata.negative_kps
                    negative_kps_xyc = clip_keypoints_to_image(negative_kps_xyc, (w, h))
                    negative_kps_xyc = rescale_keypoints(negative_kps_xyc, (w, h), (fig_w, fig_h))

                    if mode == "keypoints":
                        # compute human parsing heatmaps as gaussian on each visible keypoint
                        masks_gt_crop = build_keypoints_heatmaps(
                            keypoints_xyc, mask_w, mask_h
                        )
                    elif mode == "keypoints_gaussian":
                        # compute human parsing heatmaps as gaussian on each visible keypoint
                        masks_gt_crop = build_keypoints_gaussian_heatmaps(
                            keypoints_xyc, mask_w, mask_h, gaussian=gaussian
                        )
                    elif mode == "joints":
                        # compute human parsing heatmaps as shapes around on each visible keypoint
                        masks_gt_crop = build_joints_heatmaps(
                            keypoints_xyc, mask_w, mask_h
                        )
                    elif mode == "joints_gaussian":
                        # compute human parsing heatmaps as shapes around on each visible keypoint
                        masks_gt_crop = build_joints_gaussian_heatmaps(
                            keypoints_xyc, mask_w, mask_h
                        )
                    elif mode == "pose_on_img_crops":
                        # compute human parsing heatmaps using output of pose model on cropped person image
                        pim_img_crop = Image.fromarray(cv2.cvtColor(img_crop, cv2.COLOR_BGR2RGB))
                        processed_image, anns, meta = self.pose_model.pifpaf_preprocess(pim_img_crop, [], {})  # FIXME size
                        processed_image = processed_image.unsqueeze(0)
                        _, fields_batch = self.pose_model.processor.batch(
                            self.pose_model.model, processed_image, device=self.pose_model.device
                        )
                        masks_gt_crop = torch.concat(
                            (
                                fields_batch[0][0][:, 1],
                                fields_batch[0][1][:, 1],
                            )
                        )
                        masks_gt_crop = masks_gt_crop.unsqueeze(0)

                        masks_gt_crop = F.interpolate(
                            masks_gt_crop,
                            size=(mask_h, mask_w),
                            mode="bilinear",
                            align_corners=True
                        )

                        masks_gt_crop = masks_gt_crop.squeeze().numpy()
                        kernel = np.ones((10, 10), np.uint8)
                        if self.enable_sam:
                            # pifpaf body part masks are too coarse (overlap background) and cover all humans in
                            # the bbox. Compute a SAM segmentation mask with the pifpaf keypoints of the target person
                            # as prompt, and only keep pif and paf field inside that SAM ask.
                            sam_mask = self.compute_sam_mask(predictor, img_crop, keypoints_xyc_crop, negative_kps_xyc)
                            sam_mask = cv2.dilate(sam_mask.astype(np.uint8), kernel, iterations=2)

                            sam_mask = cv2.resize(sam_mask.squeeze(), (mask_w, mask_h))
                            #
                            masks_gt_crop = masks_gt_crop * sam_mask
                    elif mode == "pose_on_img":
                        # compute human parsing heatmaps using output of pose model on full image
                        bbox_ltwh = clip_bbox_ltwh_to_img_dim(
                            det_metadata.bbox_ltwh, img.shape[1], img.shape[0]
                        ).astype(int)
                        l, t, w, h = bbox_ltwh
                        img_crop = img[t : t + h, l : l + w]
                        img_crop = cv2.resize(img_crop, (fig_w, fig_h), cv2.INTER_CUBIC)
                        masks_gt_crop = masks_gt[:, t : t + h, l : l + w]
                        masks_gt_crop = resize(
                            masks_gt_crop, (masks_gt_crop.shape[0], fig_h, fig_w)
                        )
                        sam_mask = self.compute_sam_mask(predictor, img_crop, keypoints_xyc_crop, negative_kps_xyc)
                        masks_gt_crop = masks_gt_crop * sam_mask
                    else:
                        raise ValueError("Invalid human parsing method '{}'".format(mode))

                    # save human parsing heatmaps on disk
                    pid = det_metadata.person_id
                    filename = "{}_{}_{}".format(pid, video_id, image_id)
                    rel_filepath = Path(video_id, filename + self.masks_ext)
                    abs_filepath = Path(
                        masks_save_path, rel_filepath
                    )
                    abs_filepath.parent.mkdir(parents=True, exist_ok=True)
                    np.save(str(abs_filepath), masks_gt_crop)

                    # save image crop with human parsing heatmaps overlayed on disk for visualization/debug purpose
                    img_with_heatmap = colored_body_parts_overlay(
                        img_crop, masks_gt_crop
                    )
                    figure_filepath = Path(
                        fig_save_path, video_id, filename + "_heatmaps_" + self.img_ext
                    )
                    figure_filepath.parent.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(figure_filepath), img_with_heatmap)
                    keypoints_xyck_crop = kp_grouping_eight_bp.apply_to_keypoints_xyc(keypoints_xyc_crop)
                    img_crop_kps = draw_keypoints(img_crop, keypoints_xyck_crop, (fig_w, fig_h), radius=2, thickness=2)
                    for negative_kps in negative_kps_xyc:
                        negative_kps_xyck = kp_grouping_eight_bp.apply_to_keypoints_xyc(negative_kps)
                        img_crop_kps = draw_keypoints(img_crop_kps, negative_kps_xyck, (fig_w, fig_h), radius=2, thickness=2, color=(0, 0, 255))
                    kps_filepath = Path(
                        fig_save_path, video_id, filename + "_kps_"  + self.img_ext
                    )
                    cv2.imwrite(str(kps_filepath), img_crop_kps)
                    # record human parsing metadata for later json dump
                    gt_dets.at[det_metadata.Index, "masks_path"] = str(rel_filepath)
                    pbar.update(1)

        log.info(
            'Saving reid human parsing annotations as json to "{}"'.format(
                reid_anns_filepath
            )
        )
        reid_anns_filepath.parent.mkdir(parents=True, exist_ok=True)
        gt_dets[["id", "masks_path"]].to_json(reid_anns_filepath)

    def compute_sam_mask(self, predictor, img_crop, keypoints_xyc_crop, neg_kps_xyc):
        predictor.set_image(img_crop, image_format="BGR")
        keypoints_xyc_crop = keypoints_xyc_crop[keypoints_xyc_crop[:, -1] > 0]
        neg_kps_xyc = neg_kps_xyc.reshape((-1, 3))
        neg_kps_xyc = neg_kps_xyc[neg_kps_xyc[:, -1] > 0]
        all_keypoints = np.concatenate((keypoints_xyc_crop, neg_kps_xyc))
        keypoints_labels = np.array([1] * len(keypoints_xyc_crop) + [0] * len(neg_kps_xyc))
        sam_mask, _, _ = predictor.predict(point_coords=all_keypoints[:, :2], point_labels=keypoints_labels,
                                           multimask_output=False)
        return sam_mask

    def rescale_and_filter_keypoints(self, keypoints, bbox_ltwh, new_w, new_h):
        l, t, w, h = bbox_ltwh.astype(int)
        discarded_keypoints = 0
        rescaled_keypoints = {}
        for i, kp in enumerate(keypoints):
            # remove unvisible keypoints
            if kp[2] == 0:
                continue

            # put keypoints in bbox coord space
            kpx, kpy = kp[:2].astype(int) - np.array([l, t])

            # remove keypoints out of bbox
            if kpx < 0 or kpx >= w or kpy < 0 or kpy >= h:
                discarded_keypoints += 1
                continue

            # put keypoints in resized image coord space
            kpx, kpy = kpx * new_w / w, kpy * new_h / h

            rescaled_keypoints[i] = np.array([int(kpx), int(kpy), 1])
        return rescaled_keypoints, discarded_keypoints

    def query_gallery_split(self, gt_dets, ratio):
        def random_tracklet_sampling(_df):
            x = list(_df.index)
            size = ceil(len(x) * ratio)
            result = list(np.random.choice(x, size=size, replace=False))
            return _df.loc[result]

        def occlusion_tracklet_sampling(_df):
            _df = _df.sort_values(by=['occ_level'], ascending=False)
            indices = list(_df.index)
            result = indices[:int(len(indices) * ratio)]
            return _df.loc[result]

        gt_dets_for_reid = gt_dets[(gt_dets.split != "none")]
        # reset seed to make sure the same split is used if the dataset is instantiated multiple times
        np.random.seed(0)
        sampling = occlusion_tracklet_sampling if self.occluded_dataset else random_tracklet_sampling
        queries_per_pid = gt_dets_for_reid.groupby("person_id").apply(
            sampling
        )
        if self.eval_metric == 'mot_inter_video' or self.multi_video_queries_only:
            # keep only queries that are in more than one video
            queries_per_pid = queries_per_pid.droplevel(level=0).groupby("person_id")['video_id'].filter(lambda g: (g.nunique() > 1)).reset_index()
            assert len(queries_per_pid) != 0, "There were no identity with more than one videos to be used as queries. " \
                                              "Try setting 'multi_video_queries_only' to False or not using " \
                                              "eval_metric='mot_inter_video' or adjust the settings to sample a " \
                                              "bigger ReID dataset."
        gt_dets.loc[gt_dets.split != "none", "split"] = "gallery"
        gt_dets.loc[gt_dets.id.isin(queries_per_pid.id), "split"] = "query"

    def to_torchreid_dataset_format(self, dataframes):
        results = []
        for df in dataframes:
            df = df.copy()  # to avoid SettingWithCopyWarning
            # use video id as camera id: camid is used at inference to filter out gallery samples given a query sample
            df["camid"] = pd.Categorical(df.video_id, categories=df.video_id.unique()).codes
            df["img_path"] = df["reid_crop_path"]
            df["keypoints_xyc"] = df.apply(lambda r: kp_img_to_kp_bbox(r.keypoints_xyc, r.bbox_ltwh), axis=1)
            df["keypoints_xyc"] = df.apply(lambda r: rescale_keypoints(r.keypoints_xyc, (r.bbox_ltwh[2], r.bbox_ltwh[3]), (r.reid_crop_width, r.reid_crop_height)), axis=1)
            df["negative_kps"] = df.apply(lambda r: rescale_keypoints(r.negative_kps, (r.bbox_ltwh[2], r.bbox_ltwh[3]), (r.reid_crop_width, r.reid_crop_height)), axis=1)

            # remove bbox_head as it is not available for each sample
            # df to list of dict
            sorted_df = df.sort_values(by=["pid"])
            # use only necessary annotations: using them all caused a
            # 'RuntimeError: torch.cat(): input types can't be cast to the desired output type Long' in collate.py
            # -> still has to be fixed
            data_list = sorted_df[
                ["pid", "camid", "video_id", "img_path", "masks_path", "visibility", "keypoints_xyc", "reid_crop_width", "reid_crop_height", "negative_kps", "occ_level"]
            ]
            data_list = data_list.to_dict("records")
            results.append(data_list)
        return results

    def add_pid_column(self, gt_dets):
        # create pids as 0-based increasing numbers
        gt_dets["pid"] = None
        gt_dets_for_reid = gt_dets[(gt_dets.split != "none")]
        gt_dets.loc[gt_dets_for_reid.index, "pid"] = pd.factorize(
            gt_dets_for_reid.person_id
        )[0]

    def add_occlusion_level_column(self, gt_dets):
        def compute_occlusion_score(r):
            if r.keypoints_xyc[..., 2].sum() == 0:
                return r.negative_kps[..., 2].sum() * 2
            return r.negative_kps[..., 2].sum() / r.keypoints_xyc[..., 2].sum()
        gt_dets["occ_level"] = gt_dets.apply(compute_occlusion_score, axis=1)

    def uniform_tracklet_sampling(self, _df, max_samples_per_id, column):
        _df.sort_values(column)
        num_det = len(_df)
        if num_det > max_samples_per_id:
            # Select 'max_samples_per_id' evenly spaced indices, including first and last
            indices = np.round(np.linspace(0, num_det - 1, max_samples_per_id)).astype(
                int
            )
            assert len(indices) == max_samples_per_id
            return _df.iloc[indices]
        else:
            return _df


class SetsDict(dict):
    def __getitem__(self, key):
        if key not in self:
            raise KeyError(f"Trying to access a '{key}' split of the dataset that is not available. "
                           f"Available splits are {list(self.keys())}. "
                           f"Make sur this split name is correct or is available in the dataset folder.")
        return super().__getitem__(key)


@dataclass
class TrackingSet:
    video_metadatas: pd.DataFrame
    image_metadatas: pd.DataFrame
    detections_gt: pd.DataFrame
    image_gt: pd.DataFrame = pd.DataFrame(columns=["video_id"])


class TrackingDataset(ABC):
    def __init__(
        self,
        dataset_path: str,
        sets: dict[str, TrackingSet],
        nvid: int = -1,
        nframes: int = -1,
        vids_dict: list = None,
        *args,
        **kwargs
    ):
        self.dataset_path = Path(dataset_path)
        self.sets = SetsDict(sets)
        sub_sampled_sets = SetsDict()
        for set_name, split in self.sets.items():
            vid_list = vids_dict[set_name] if vids_dict is not None and set_name in vids_dict else None
            sub_sampled_sets[set_name] = self._subsample(split, nvid, nframes, vid_list)
        self.sets = sub_sampled_sets

    def _subsample(self, tracking_set, nvid, nframes, vids_names):
        if nvid < 1 and nframes < 1 and (vids_names is None or len(vids_names) == 0) or tracking_set is None:
            return tracking_set

        # filter videos:
        if vids_names is not None and len(vids_names) > 0:
            assert set(vids_names).issubset(tracking_set.video_metadatas.name.unique()), f"Some videos to process {set(vids_names) - set(tracking_set.video_metadatas.name.unique())} does not exist in the tracking set"
            videos_to_keep = tracking_set.video_metadatas[
                tracking_set.video_metadatas.name.isin(vids_names)
            ].index
            tiny_video_metadatas = tracking_set.video_metadatas.loc[videos_to_keep]
        elif nvid > 0:  # keep 'nvid' videos
            videos_to_keep = tracking_set.video_metadatas.sample(
                nvid, random_state=2
            ).index
            tiny_video_metadatas = tracking_set.video_metadatas.loc[videos_to_keep]
        else:  # keep all videos
            videos_to_keep = tracking_set.video_metadatas.index
            tiny_video_metadatas = tracking_set.video_metadatas

        # filter images:
        # keep only images from videos to keep
        tiny_image_metadatas = tracking_set.image_metadatas[
            tracking_set.image_metadatas.video_id.isin(videos_to_keep)
        ]
        tiny_image_gt = tracking_set.image_gt[
            tracking_set.image_gt.video_id.isin(videos_to_keep)
        ]

        # keep only images from first nframes
        if nframes > 0:
            tiny_image_metadatas = tiny_image_metadatas.groupby("video_id").head(
                nframes
            )
            tiny_image_gt = tiny_image_gt.groupby("video_id").head(nframes)

        # filter detections:
        tiny_detections = None
        if tracking_set.detections_gt is not None and not tracking_set.detections_gt.empty:
            tiny_detections = tracking_set.detections_gt[
                tracking_set.detections_gt.image_id.isin(tiny_image_metadatas.index)
            ]

        assert len(tiny_video_metadatas) > 0, "No videos left after subsampling the tracking set"
        assert len(tiny_image_metadatas) > 0, "No images left after subsampling the tracking set"

        return TrackingSet(
            tiny_video_metadatas,
            tiny_image_metadatas,
            tiny_detections,
            tiny_image_gt,
        )


    @staticmethod
    def _mot_encoding(detections, image_metadatas, video_metadatas, bbox_column):
        detections = detections.copy()
        image_metadatas["id"] = image_metadatas.index
        df = pd.merge(
            image_metadatas.reset_index(drop=True),
            detections.reset_index(drop=True),
            left_on="id",
            right_on="image_id",
            suffixes=('', '_y')
        )
        len_before_drop = len(df)
        df.dropna(
            subset=[
                "frame",
                "track_id",
                bbox_column,
            ],
            how="any",
            inplace=True,
        )

        if len_before_drop != len(df):
            log.warning(
                "Dropped {} rows with NA values".format(len_before_drop - len(df))
            )
        df["track_id"] = df["track_id"].astype(int)
        df["bb_left"] = df[bbox_column].apply(lambda x: x[0])
        df["bb_top"] = df[bbox_column].apply(lambda x: x[1])
        df["bb_width"] = df[bbox_column].apply(lambda x: x[2])
        df["bb_height"] = df[bbox_column].apply(lambda x: x[3])
        df = df.assign(x=-1, y=-1, z=-1)
        return df


    def save_for_eval(self,
                      detections: pd.DataFrame,
                      image_metadatas: pd.DataFrame,
                      video_metadatas: pd.DataFrame,
                      save_folder: str,
                      bbox_column_for_eval="bbox_ltwh",
                      save_classes=False,
                      is_ground_truth=False,
                      save_zip=True
                      ):
        """Save predictions in MOT Challenge format."""
        mot_df = self._mot_encoding(detections, image_metadatas, video_metadatas, bbox_column_for_eval)

        save_path = os.path.join(save_folder)
        os.makedirs(save_path, exist_ok=True)

        # MOT Challenge format = <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <x>, <y>, <z>
        # videos_names = mot_df["video_name"].unique()
        for id, video in video_metadatas.iterrows():
            file_path = os.path.join(save_path, f"{video['name']}.txt")
            file_df = mot_df[mot_df["video_id"] == id].copy()
            if file_df["frame"].min() == 0:
                file_df["frame"] = file_df["frame"] + 1  # MOT Challenge format starts at 1
            if not file_df.empty:
                file_df.sort_values(by="frame", inplace=True)
                clazz = "category_id" if save_classes else "x"
                file_df[
                    [
                        "frame",
                        "track_id",
                        "bb_left",
                        "bb_top",
                        "bb_width",
                        "bb_height",
                        "bbox_conf",
                        clazz,
                        "y",
                        "z",
                    ]
                ].to_csv(
                    file_path,
                    header=False,
                    index=False,
                )
            else:
                open(file_path, "w").close()

    def process_trackeval_results(self, results, dataset_config, eval_config):
        log.info(f"TrackEval results = {results}")
        return results

available_keypoints = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]


class PoseTrack21(TrackingDataset):
    """
    Train set: 43603 images
    Val set: 20161 images
    Test set: ??? images
    """

    def __init__(
        self,
        dataset_path: str,
        annotation_path: str,
        posetrack_version=21,
        *args,
        **kwargs
    ):
        self.dataset_path = Path(dataset_path)
        assert self.dataset_path.exists(), "'{}' directory does not exist. Either put the dataset under this path or change the dataset path config under 'config.data.root' (just specify the root folder of PoseTrack21, 'PoseTrack21' folder name will be concatenated to it automatically). ".format(
            self.dataset_path
        )
        self.annotation_path = Path(annotation_path)
        assert self.annotation_path.exists(), "'{}' directory does not exist".format(
            self.annotation_path
        )

        train_set = load_tracking_set(
            self.annotation_path / "train", self.dataset_path, posetrack_version
        )
        val_set = load_tracking_set(
            self.annotation_path / "val", self.dataset_path, posetrack_version
        )
        test_set = None  # TODO

        sets = {"train": train_set, "val": val_set, "test": test_set}

        super().__init__(dataset_path, sets, *args, **kwargs)


def load_tracking_set(anns_path, dataset_path, posetrack_version=21):
    # Load annotations into Pandas dataframes
    video_metadatas, image_metadatas, detections_gt = load_annotations(anns_path)
    # Fix formatting of dataframes to be compatible with tracklab
    video_metadatas, image_metadatas, detections_gt = fix_formatting(
        video_metadatas, image_metadatas, detections_gt, dataset_path, posetrack_version
    )
    return TrackingSet(
        video_metadatas,
        image_metadatas,
        detections_gt,
    )


def load_annotations(anns_path):
    anns_files_list = list(anns_path.glob("*.json"))
    assert len(anns_files_list) > 0, "No annotations files found in {}".format(
        anns_path
    )
    detections_gt = []
    image_metadatas = []
    video_metadatas = []
    for path in anns_files_list:
        with open(path) as json_file:
            data_dict = json.load(json_file)
            detections_gt.extend(data_dict["annotations"])
            image_metadatas.extend(data_dict["images"])
            video_metadatas.append(
                {
                    "id": data_dict["images"][0]["vid_id"],
                    "nframes": len(data_dict["images"]),
                    "name": path.stem,
                    "categories": data_dict["categories"],
                }
            )

    return (
        pd.DataFrame(video_metadatas),
        pd.DataFrame(image_metadatas),
        pd.DataFrame(detections_gt),
    )


def fix_formatting(
    video_metadatas, image_metadatas, detections_gt, dataset_path, posetrack_version
):
    image_id = "image_id" if posetrack_version == 21 else "frame_id"

    # Videos
    video_metadatas.set_index("id", drop=False, inplace=True)

    # Images
    image_metadatas.drop([image_id, "nframes"], axis=1, inplace=True)
    image_metadatas["file_name"] = image_metadatas["file_name"].apply(
        lambda x: os.path.join(dataset_path, x)
    )
    image_metadatas["frame"] = image_metadatas["file_name"].apply(
        lambda x: int(os.path.basename(x).split(".")[0]) + 1
    )
    image_metadatas.rename(
        columns={"vid_id": "video_id", "file_name": "file_path"},
        inplace=True,
    )
    image_metadatas.set_index("id", drop=False, inplace=True)

    # Detections
    detections_gt.drop(["bbox_head"], axis=1, inplace=True)
    detections_gt.rename(columns={"bbox": "bbox_ltwh"}, inplace=True)
    detections_gt.bbox_ltwh = detections_gt.bbox_ltwh.apply(lambda x: np.array(x))
    detections_gt.rename(columns={"keypoints": "keypoints_xyc"}, inplace=True)
    detections_gt.keypoints_xyc = detections_gt.keypoints_xyc.apply(
        lambda x: np.reshape(np.array(x), (-1, 3))
    )
    detections_gt.set_index("id", drop=False, inplace=True)
    # compute detection visiblity as average keypoints visibility
    detections_gt["visibility"] = detections_gt.keypoints_xyc.apply(lambda x: x[available_keypoints, 2].mean())
    detections_gt = detections_gt.merge(
        image_metadatas[["video_id"]], how="left", left_on="image_id", right_index=True
    )

    return video_metadatas, image_metadatas, detections_gt


class Module(metaclass=ABCMeta):
    input_columns = None
    output_columns = None
    training_enabled = False
    forget_columns = []

    @property
    def name(self):
        name = self.__class__.__name__
        return name  # re.sub("([a-z0-9])([A-Z])", r"\1_\2", name).lower()

    @property
    def level(self):
        name = self.__class__.__bases__[0].__name__
        name = re.sub("([a-z0-9])([A-Z])", r"\1_\2", name).lower()
        return name.split("_")[0]

    def validate_input(self, dataframe):
        assert self.input_columns is not None, "Every model should define its inputs"
        for col in self.input_columns:
            if col not in dataframe.columns:
                raise AttributeError(f"The input detection should contain {col}.")

    def validate_output(self, dataframe):
        assert self.output_columns is not None, "Every model should define its outputs"
        for col in self.output_columns:
            if col not in dataframe.columns:
                raise AttributeError(f"The output detection should contain {col}.")

    def get_input_columns(self, level):
        if isinstance(self.input_columns, list):
            return self.input_columns if level == "detection" else []
        elif isinstance(self.input_columns, dict):
            return self.input_columns.get(level, [])

    def get_output_columns(self, level):
        if isinstance(self.output_columns, list):
            return self.output_columns if level == "detection" else []
        elif isinstance(self.output_columns, dict):
            return self.output_columns.get(level, [])


class ImageLevelModule(Module):
    """Abstract class to implement a module that operates directly on images.

    This can for example be a bounding box detector, or a bottom-up
    pose estimator (which outputs keypoints directly).

    The functions to implement are
     - __init__, which can take any configuration needed
     - preprocess
     - process
     - datapipe (optional) : returns an object which will be used to create the pipeline.
                            (Only modify this if you know what you're doing)
     - dataloader (optional) : returns a dataloader for the datapipe

     You should also provide the following class properties :
      - input_columns : what info you need for the detections
      - output_columns : what info you will provide when called
      - collate_fn (optional) : the function that will be used for collating the inputs
                                in a batch. (Default : pytorch collate function)

     A description of the expected behavior is provided below.
    """

    collate_fn = default_collate
    input_columns = None
    output_columns = None

    @abstractmethod
    def __init__(self, batch_size: int):
        """Init function

        The arguments to this function are completely free
        and will be provided by a configuration file.

        You should call the __init__ function from the super() class.
        """
        self.batch_size = batch_size
        self._datapipe = None

    @abstractmethod
    def preprocess(self, image, detections: pd.DataFrame, metadata: pd.Series) -> Any:
        """Adapts the default input to your specific case.

        Args:
            image: a numpy array of the current image
            detections: a DataFrame containing all the detections pertaining to a single
                        image
            metadata: additional information about the image

        Returns:
            preprocessed_sample: input for the process function
        """
        pass

    @abstractmethod
    def process(self, batch: Any, detections: pd.DataFrame, metadatas: pd.DataFrame):
        """The main processing function. Runs on GPU.

        Args:
            batch: The batched outputs of `preprocess`
            detections: The previous detections.
            metadatas: The previous image metadatas

        Returns:
            output : Either a DataFrame containing the new/updated detections
                    or a tuple containing detections and metadatas (in that order)
                    The DataFrames can be either a list of Series, a list of DataFrames
                    or a single DataFrame. The returned objects will be aggregated
                    automatically according to the `name` of the Series/`index` of
                    the DataFrame. **It is thus mandatory here to name correctly
                    your series or index your dataframes.**
                    The output will override the previous detections
                    with the same name/index.
        """
        pass

    @property
    def datapipe(self):
        if self._datapipe is None:
            self._datapipe = EngineDatapipe(self)
        return self._datapipe

    def dataloader(self, engine: "TrackingEngine"):
        datapipe = self.datapipe
        return DataLoader(
            dataset=datapipe,
            batch_size=self.batch_size,
            collate_fn=type(self).collate_fn,
            num_workers=engine.num_workers,
            persistent_workers=False,
        )


def collate_images_anns_meta(batch):
    idxs = [b[0] for b in batch]
    batch = [b[1] for b in batch]
    anns = [b[-2] for b in batch]
    metas = [b[-1] for b in batch]

    processed_images = torch.utils.data.dataloader.default_collate(
        [b[0] for b in batch]
    )
    idxs = torch.utils.data.dataloader.default_collate(idxs)
    return idxs, (processed_images, anns, metas)


class OpenPifPaf:
    collate_fn = collate_images_anns_meta

    def __init__(self, cfg, device):
        self.cfg = cfg
        self.device = device
        self.id = 0

        if cfg.predict.checkpoint:
            import openpifpaf
            old_argv = sys.argv
            sys.argv = self._hydra_to_argv(cfg.predict)
            openpifpaf.predict.pbtrack_cli()
            predictor = openpifpaf.Predictor()
            sys.argv = old_argv

            self.model = predictor.model
            self.pifpaf_preprocess = predictor.preprocess
            self.processor = predictor.processor
            log.info(
                f"Loaded detection model from checkpoint: {cfg.predict.checkpoint}"
            )

    def _hydra_to_argv(self, cfg):
        new_argv = ["argv_from_hydra"]
        for k, v in cfg.items():
            new_arg = f"--{str(k)}"
            if isinstance(v, ListConfig):
                new_argv.append(new_arg)
                for item in v:
                    new_argv.append(f"{str(item)}")
            elif v is not None:
                new_arg += f"={str(v)}"
                new_argv.append(new_arg)
            else:
                new_argv.append(new_arg)
        return new_argv


class EngineDatapipe(Dataset):
    def __init__(self, model) -> None:
        self.model = model
        self.image_filepaths = None
        self.img_metadatas = None
        self.detections = None

    def update(self, image_filepaths: dict, img_metadatas, detections):
        del self.img_metadatas
        del self.detections
        self.image_filepaths = image_filepaths
        self.img_metadatas = img_metadatas
        self.detections = detections

    def __len__(self):
        if self.model.level == "detection":
            return len(self.detections)
        elif self.model.level == "image":
            return len(self.img_metadatas)
        else:
            raise ValueError(f"You should provide the appropriate level for you module not '{self.model.level}'")

    def __getitem__(self, idx):
        if self.model.level == "detection":
            detection = self.detections.iloc[idx]
            metadata = self.img_metadatas.loc[detection.image_id]
            image = cv2_load_image(self.image_filepaths[metadata.name])
            sample = (
                detection.name,
                self.model.preprocess(image=image, detection=detection, metadata=metadata),
            )
            return sample
        elif self.model.level == "image":
            metadata = self.img_metadatas.iloc[idx]
            if self.detections is not None and len(self.detections) > 0:
                detections = self.detections[self.detections.image_id == metadata.name]
            else:
                detections = self.detections
            image = cv2_load_image(self.image_filepaths[metadata.name])
            sample = (self.img_metadatas.index[idx], self.model.preprocess(
                image=image, detections=detections, metadata=metadata))
            return sample
        else:
            raise ValueError("Please provide appropriate level.")

def generate_bbox_from_keypoints(keypoints, extension_factor, image_shape=None):
    """
    Generates a bounding box from keypoints by computing the bounding box of the keypoints and extending it by a factor.

    Args:
        keypoints (np.ndarray): A numpy array of shape (K, 3) representing the keypoints in the format (x, y, c).
        extension_factor (tuple): A tuple of float [top, bottom, right&left] representing the factor by which
        the bounding box should be extended based on the keypoints.
        image_shape (tuple): A tuple of two integers representing the image dimensions `(width, height)`.

    Returns:
        np.ndarray: A numpy array of shape (4,) representing the bounding box in the format (left, top, w, h).
    """
    keypoints = sanitize_keypoints(keypoints, image_shape)
    keypoints = keypoints[keypoints[:, 2] > 0]
    lt, rb = np.min(keypoints[:, :2], axis=0), np.max(keypoints[:, :2], axis=0)
    w, h = rb - lt
    lt -= np.array([extension_factor[2] * w, extension_factor[0] * h])
    rb += np.array([extension_factor[2] * w, extension_factor[1] * h])
    bbox = np.concatenate([lt, rb - lt])
    bbox = sanitize_bbox_ltwh(bbox, image_shape)
    return bbox


def sanitize_keypoints(keypoints, image_shape=None, rounded=False):
    """
    Sanitizes keypoints by clipping them to the image dimensions and ensuring that their confidence values are valid.

    Args:
        keypoints (np.ndarray): A numpy array of shape (K, 2 or 3) representing the keypoints in the format (x, y, (c)).
        image_shape (tuple): A tuple of two integers representing the image dimensions `(width, height)`.
        rounded (bool): Whether to round the keypoints to integers.

    Returns:
        np.ndarray: A numpy array of shape (K, 3) representing the sanitized keypoints in the format (x, y, (c)).
    """
    assert isinstance(keypoints, np.ndarray), "Keypoints must be a numpy array."
    assert keypoints.ndim == 2 and keypoints.shape[1] in (
        2,
        3,
    ), "Keypoints must be a numpy array of shape (K, 2 or 3)."
    if image_shape is not None:
        keypoints[:, 0] = np.clip(keypoints[:, 0], 0, image_shape[0] - 1)
        keypoints[:, 1] = np.clip(keypoints[:, 1], 0, image_shape[1] - 1)
    if rounded:
        keypoints[:, :2] = np.round(keypoints[:, :2]).astype(int)
    return keypoints


def clip_bbox_ltwh_to_img_dim(bbox_ltwh, img_w, img_h):
    """
    Clip bounding box to image dimensions.
    Args:
        bbox_ltwh (np.ndarray): bounding box, shape (4,)
        img_w (int): image width
        img_h (int): image height
    Returns:
        bbox_ltwh (np.ndarray): clipped bounding box, shape (4,)
    """
    l, t, w, h = bbox_ltwh
    l = np.clip(l, 0, img_w - 1)
    t = np.clip(t, 0, img_h - 1)
    w = np.clip(w, 0, img_w - 1 - l)
    h = np.clip(h, 0, img_h - 1 - t)
    return np.array([l, t, w, h])


def clip_keypoints_to_image(kps, image_size):
    """
    Clip keypoints to image size.

    Parameters:
    - kps: a tensor/array of size 17x3 representing keypoints.
           Can be either a numpy array or a torch tensor.
    - image_size (tuple): a tuple containing the width and height of the target image.

    Returns:
    - clipped_kps: keypoints clipped to image size.
                   Returns in the same format as input (numpy array or torch tensor).
    """

    # Get image width and height
    w, h = image_size

    # Check if the input is a numpy array
    if isinstance(kps, np.ndarray):
        kps[..., 0] = np.clip(kps[..., 0], 0, w)
        kps[..., 1] = np.clip(kps[..., 1], 0, h)
        return kps
    # Check if the input is a torch tensor
    elif torch.is_tensor(kps):
        kps[..., 0] = torch.clamp(kps[..., 0], 0, w)
        kps[..., 1] = torch.clamp(kps[..., 1], 0, h)
        return kps
    else:
        raise ValueError("Input keypoints must be either a numpy array or a torch tensor.")


def keypoints_in_bbox_coord(kp_xyc_img, bbox_ltwh):
    """
    Convert keypoints in image coordinates to bounding box coordinates and filter out keypoints that are outside the
    bounding box.
    Args:
        kp_xyc_img (np.ndarray): keypoints in image coordinates, shape (K, 2)
        bbox_tlwh (np.ndarray): bounding box, shape (4,)
    Returns:
        kp_xyc_bbox (np.ndarray): keypoints in bounding box coordinates, shape (K, 2)
    """
    l, t, w, h = bbox_ltwh
    kp_xyc_bbox = kp_xyc_img.copy()

    # put keypoints in bbox coord space
    kp_xyc_bbox[..., 0] = kp_xyc_img[..., 0] - l
    kp_xyc_bbox[..., 1] = kp_xyc_img[..., 1] - t

    # remove out of bbox keypoints
    kp_xyc_bbox[
        (kp_xyc_bbox[..., 2] == 0)
        | (kp_xyc_bbox[..., 0] < 0)
        | (kp_xyc_bbox[..., 0] >= w)
        | (kp_xyc_bbox[..., 1] < 0)
        | (kp_xyc_bbox[..., 1] >= h)
    ] = 0

    return kp_xyc_bbox


def cv2_load_image(file_path):
    file_path = str(file_path)
    image = cv2.imread(str(file_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def sanitize_bbox_ltwh(bbox: np.array, image_shape=None, rounded=False):
    """
    Sanitizes a bounding box by clipping it to the image dimensions and ensuring that its dimensions are valid.

    Args:
        bbox (np.ndarray): A numpy array of shape (4,) representing the bounding box in the format
        `[left, top, width, height]`.
        image_shape (tuple): A tuple of two integers representing the image dimensions `(width, height)`.
        rounded (bool): Whether to round the bounding box coordinates, type becomes int.

    Returns:
        np.ndarray: A numpy array of shape (4,) representing the sanitized bounding box in the format
        `[left, top, width, height]`.
    """
    assert isinstance(
        bbox, np.ndarray
    ), f"Expected bbox to be of type np.ndarray, got {type(bbox)}"
    assert bbox.shape == (4,), f"Expected bbox to be of shape (4,), got {bbox.shape}"
    if image_shape is not None:
        bbox[0] = max(0, min(bbox[0], image_shape[0] - 2))
        bbox[1] = max(0, min(bbox[1], image_shape[1] - 2))
        bbox[2] = max(1, min(bbox[2], image_shape[0] - 1 - bbox[0]))
        bbox[3] = max(1, min(bbox[3], image_shape[1] - 1 - bbox[1]))
    if rounded:
        bbox = bbox.round().astype(int)
    return bbox
