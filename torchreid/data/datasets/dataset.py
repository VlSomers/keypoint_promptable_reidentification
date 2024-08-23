from __future__ import division, print_function, absolute_import
import copy
import os

import math
import numpy as np
import os.path as osp
import tarfile
import zipfile
import torch
from pathlib import Path
from copy import deepcopy

from torchreid.data.data_augmentation import load_human_occluders
from torchreid.data.datasets.keypoints_to_masks import KeypointsToMasks
from torchreid.utils import read_masks, read_image, download_url, mkdir_if_missing
from torchreid.utils.tools import read_keypoints


class Dataset(object):
    """An abstract class representing a Dataset.

    This is the base class for ``ImageDataset`` and ``VideoDataset``.

    Args:
        train (list): contains tuples of (img_path(s), pid, camid).
        query (list): contains tuples of (img_path(s), pid, camid).
        gallery (list): contains tuples of (img_path(s), pid, camid).
        transform: transform function.
        mode (str): 'train', 'query' or 'gallery'.
        combineall (bool): combines train, query and gallery in a
            dataset for training.
        verbose (bool): show information.
    """
    _junk_pids = [
    ] # contains useless person IDs, e.g. background, false detections

    masks_base_dir = None
    eval_metric = 'default'  # default to market101
    cam_num = 0
    view = 0

    def gallery_filter(self, q_pid, q_camid, q_ann, g_pids, g_camids, g_anns):
        """ Remove gallery samples that have the same pid and camid as the query sample, since ReID is a cross-camera
        person retrieval task for most datasets. However, we still keep samples from the same camera but of different
        identity as distractors."""
        remove = (g_camids == q_camid) & (g_pids == q_pid)
        return remove

    def infer_masks_path(self, img_path, masks_dir, masks_suffix):
        masks_path = os.path.join(self.dataset_dir, self.masks_base_dir, masks_dir, os.path.basename(os.path.dirname(img_path)), os.path.splitext(os.path.basename(img_path))[0] + masks_suffix)
        return masks_path

    def infer_kp_path(self, img_path):
        masks_path = os.path.join(self.dataset_dir, 'external_annotation', self.kp_dir, os.path.basename(os.path.dirname(img_path)), os.path.splitext(os.path.basename(img_path))[0] + '.jpg_keypoints.json')
        return masks_path

    def __init__(
        self,
        train,
        query,
        gallery,
        config=None,
        transform_tr=None,
        transform_te=None,
        kp_target_transform=None,
        kp_prompt_transform=None,
        mode='train',
        combineall=False,
        verbose=True,
        masks_dir=None,
        masks_base_dir=None,
        load_masks=False,
        random_occlusions=False,
        **kwargs
    ):
        self.train = train
        self.filter_out_no_skeletons = config.model.kpr.keypoints.filter_out_no_skeletons
        if self.filter_out_no_skeletons:
            or_size = len(self.train)
            self.train = [sample for sample in self.train if 'kp_path' in sample and read_keypoints(sample['kp_path']).max()>0]
            print('Filtered out {}/{} samples without keypoints'.format(or_size - len(self.train), or_size))
            assert len(self.train) > 0, 'No samples with keypoints found'
        query_set = set(config.data.query_list)
        if len(query_set) > 0:
            self.query = [q for q in query if Path(q['img_path']).stem in query_set]
        else:
            self.query = query
        self.gallery = gallery
        self.transform_tr = transform_tr
        self.transform_te = transform_te
        self.cfg = config
        self.target_preprocess = kp_target_transform
        self.prompt_preprocess = kp_prompt_transform
        self.keypoints_to_prompt_masks = KeypointsToMasks(mode=self.cfg.model.kpr.keypoints.prompt_masks,
                                                          vis_thresh=self.cfg.model.kpr.keypoints.vis_thresh,
                                                          vis_continous=self.cfg.model.kpr.keypoints.vis_continous,
                                                          )
        self.keypoints_to_target_masks = KeypointsToMasks(mode=self.cfg.model.kpr.keypoints.target_masks,
                                                          vis_thresh=self.cfg.model.kpr.keypoints.vis_thresh,
                                                          vis_continous=False,
                                                          )
        self.mode = mode
        self.combineall = combineall
        self.verbose = verbose
        self.masks_dir = masks_dir
        self.load_masks = load_masks
        if masks_base_dir is not None:
            self.masks_base_dir = masks_base_dir

        self.random_occlusions = random_occlusions
        if self.random_occlusions:
            # FIXME should take into account combining datasets, query/gallery folder, etc
            self.all_occluders = load_human_occluders(self.train, img_size=(config.data.width, config.data.height))

        self.num_train_pids = self.get_num_pids(self.train)
        self.num_train_cams = self.get_num_cams(self.train)
        if self.combineall:
            self.combine_all()

        if self.verbose:
            self.show_summary()

    # def compute_path_for_random_occlusions(self):
    #     imgs_path = Path(self.train_dir)
    #     masks_dir = self.cfg.data.bipo.masks_dir
    #     kp_dir = self.cfg.model.kpr.keypoints.kp_dir
    #     masks_path = imgs_path.parent / "masks" / masks_dir / imgs_path.name
    #     keypoints_path = imgs_path.parent / "external_annotation" / kp_dir / imgs_path.name
    #     return keypoints_path, masks_path

    def transforms(self, mode):
        """Returns the transforms of a specific mode."""
        if mode == 'train':
            return self.transform_tr
        elif mode == 'query':
            return self.transform_te
        elif mode == 'gallery':
            return self.transform_te
        else:
            raise ValueError("Invalid mode. Got {}, but expected to be "
                             "'train', 'query' or 'gallery'".format(mode))

    def data(self, mode):
        """Returns the data of a specific mode.

        Args:
            mode (str): 'train', 'query' or 'gallery'.

        Returns:
            list: contains tuples of (img_path(s), pid, camid).
        """
        if mode == 'train':
            return self.train
        elif mode == 'query':
            return self.query
        elif mode == 'gallery':
            return self.gallery
        else:
            raise ValueError("Invalid mode. Got {}, but expected to be "
                             "'train', 'query' or 'gallery'".format(mode))

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):  # kept for backward compatibility
        return self.len(self.mode)

    def len(self, mode):
        return len(self.data(mode))

    def __add__(self, other):
        """Adds two datasets together (only the train set)."""
        train = copy.deepcopy(self.train)

        for sample in other.train:
            sample['pid'] += self.num_train_pids
            train.append(sample)

        ###################################
        # Things to do beforehand:
        # 1. set verbose=False to avoid unnecessary print
        # 2. set combineall=False because combineall would have been applied
        #    if it was True for a specific dataset, setting it to True will
        #    create new IDs that should have been included
        ###################################


        if isinstance(self, ImageDataset):
            return ImageDataset(
                train,
                self.query,
                self.gallery,
                transform=self.transform,
                mode=self.mode,
                combineall=False,
                verbose=False,
                masks_base_dir=self.masks_base_dir,
            )
        else:
            return VideoDataset(
                train,
                self.query,
                self.gallery,
                transform=self.transform,
                mode=self.mode,
                combineall=False,
                verbose=False,
                seq_len=self.seq_len,
                sample_method=self.sample_method
            )

    def __radd__(self, other):
        """Supports sum([dataset1, dataset2, dataset3])."""
        if other == 0:
            return self
        else:
            return self.__add__(other)

    def parse_data(self, data):
        """Parses data list and returns the number of person IDs
        and the number of camera views.

        Args:
            data (list): contains tuples of (img_path(s), pid, camid)
        """
        pids = set()
        cams = set()
        for i, sample in enumerate(data):
            pids.add(sample['pid'])
            cams.add(sample['camid'])
        return len(pids), len(cams)

    def get_num_pids(self, data):
        """Returns the number of training person identities."""
        return self.parse_data(data)[0]

    def get_num_cams(self, data):
        """Returns the number of training cameras."""
        return self.parse_data(data)[1]

    def show_summary(self):
        """Shows dataset statistics."""
        pass

    def combine_all(self):
        """Combines train, query and gallery in a dataset for training."""
        combined = copy.deepcopy(self.train)

        # relabel pids in gallery (query shares the same scope)
        g_pids = set()
        for sample in self.gallery:
            pid = sample['pid']
            if pid in self._junk_pids:
                continue
            g_pids.add(pid)
        pid2label = {pid: i for i, pid in enumerate(g_pids)}

        def _combine_data(data):
            for sample in data:
                pid = sample['pid']
                if pid in self._junk_pids:
                    continue
                sample['pid'] = pid2label[pid] + self.num_train_pids
                combined.append(sample)

        _combine_data(self.query)
        _combine_data(self.gallery)

        self.train = combined
        self.num_train_pids = self.get_num_pids(self.train)

    def download_dataset(self, dataset_dir, dataset_url):
        """Downloads and extracts dataset.

        Args:
            dataset_dir (str): dataset directory.
            dataset_url (str): url to download dataset.
        """
        if osp.exists(dataset_dir):
            return

        if dataset_url is None:
            raise RuntimeError(
                '{} dataset needs to be manually '
                'prepared, please follow the '
                'document to prepare this dataset'.format(
                    self.__class__.__name__
                )
            )

        print('Creating directory "{}"'.format(dataset_dir))
        mkdir_if_missing(dataset_dir)
        fpath = osp.join(dataset_dir, osp.basename(dataset_url))

        print(
            'Downloading {} dataset to "{}"'.format(
                self.__class__.__name__, dataset_dir
            )
        )
        download_url(dataset_url, fpath)

        print('Extracting "{}"'.format(fpath))
        try:
            tar = tarfile.open(fpath)
            tar.extractall(path=dataset_dir)
            tar.close()
        except:
            zip_ref = zipfile.ZipFile(fpath, 'r')
            zip_ref.extractall(dataset_dir)
            zip_ref.close()

        print('{} dataset is ready'.format(self.__class__.__name__))

    def check_before_run(self, required_files):
        """Checks if required files exist before going deeper.

        Args:
            required_files (str or list): string file name(s).
        """
        if isinstance(required_files, str):
            required_files = [required_files]

        for fpath in required_files:
            if not osp.exists(fpath):
                raise RuntimeError('"{}" is not found'.format(fpath))

    def __repr__(self):
        try:
            num_train_pids, num_train_cams = self.parse_data(self.train)
            num_query_pids, num_query_cams = self.parse_data(self.query)
            num_gallery_pids, num_gallery_cams = self.parse_data(self.gallery)

            msg = '  ----------------------------------------\n' \
                  '  subset   | # ids | # items | # cameras\n' \
                  '  ----------------------------------------\n' \
                  '  train    | {:5d} | {:7d} | {:9d}\n' \
                  '  query    | {:5d} | {:7d} | {:9d}\n' \
                  '  gallery  | {:5d} | {:7d} | {:9d}\n' \
                  '  ----------------------------------------\n' \
                  '  items: images/tracklets for image/video dataset\n'.format(
                      num_train_pids, len(self.train), num_train_cams,
                      num_query_pids, len(self.query), num_query_cams,
                      num_gallery_pids, len(self.gallery), num_gallery_cams
                  )
        except:
            msg = "Non initialized dataset"
        return msg


class ImageDataset(Dataset):
    """A base class representing ImageDataset.

    All other image datasets should subclass it.

    ``__getitem__`` returns an image given index.
    It will return ``img``, ``pid``, ``camid`` and ``img_path``
    where ``img`` has shape (channel, height, width). As a result,
    data in each batch has shape (batch_size, channel, height, width).
    """

    def __init__(self, train, query, gallery, **kwargs):
        super(ImageDataset, self).__init__(train, query, gallery, **kwargs)

    def __getitem__(self, index):  # kept for backward compatibility
        if isinstance(index, tuple):
            index, batch_pids = index
        else:
            batch_pids = None
        sample = deepcopy(self.data(self.mode)[index])
        all_occluders = self.all_occluders if self.random_occlusions else None
        return self.getitem(sample, self.cfg, self.keypoints_to_prompt_masks, self.prompt_preprocess, self.keypoints_to_target_masks, self.target_preprocess, self.transforms(self.mode), self.load_masks, batch_pids, all_occluders)


    @staticmethod
    def getitem(sample, cfg, keypoints_to_prompt_masks, prompt_preprocess, keypoints_to_target_masks, target_preprocess, transforms, load_masks=False, batch_pids=None, all_occluders=None):
        """
        Complex getitem function that should be refactored and cleaned.
        This method will load the training/test samples, i.e. the reid image, the keypoints, and the human parsing labels (often called 'masks').
        Image, keypoints and masks then undergo the configured transformations (e.g. resize, crop, flip, BIPO to generate random person occlusion, ...).
        Albumentation is employed as a data augmentation library to handle coherent transformation across the three loaded values.
        Keypoints and masks are also grouped into a fixed set of body parts (often 5 or 8) here.
        List of keypoints in (x, y, c) format are converted to dense masks/heatmaps here too.
        """
        
        spatial_feature_shape = cfg.model.kpr.spatial_feature_shape
        kp_enabled = cfg.model.kpr.keypoints.enabled
        pose_encoding_strategy = cfg.model.promptable_trans.pose_encoding_strategy
        masks_prompting = cfg.model.promptable_trans.masks_prompting
        use_negative_keypoints = cfg.model.kpr.keypoints.use_negative_keypoints
        masks_enabled = cfg.model.kpr.masks.enabled
        # load_masks = cfg.model.kpr.masks.preprocess in masks_preprocess_transforms or cfg.model.kpr.masks.preprocess == 'none'

        transf_args = {}
        if "image" in sample:
            transf_args["image"] = sample["image"]
        else:
            transf_args["image"] = read_image(sample["img_path"])
        or_img_size = (transf_args["image"].shape[1], transf_args["image"].shape[0])
        transf_args["mask_size"] = (spatial_feature_shape[0], spatial_feature_shape[1])  # = (H, W)

        if all_occluders:  # needed for batch aware person copy paste to generate random occlusions
            transf_args["batch_pids"] = batch_pids
            transf_args["all_occluders"] = all_occluders

        if kp_enabled:
            if 'keypoints_xyc' in sample or 'kp_path' in sample:
                assert not (keypoints_to_prompt_masks.mode == 'none' and keypoints_to_target_masks.mode == 'none')
                if 'kp_path' in sample:  # for standard reid datasets
                    assert 'keypoints_xyc' not in sample
                    kp_path = sample['kp_path']
                    bbox_ltwh = (0, 0, transf_args["image"].shape[1], transf_args["image"].shape[0])
                    keypoints_xyc = read_keypoints(kp_path, bbox_ltwh)
                    kps_shape = keypoints_xyc.shape
                    keypoints_xyc = keypoints_xyc.reshape(keypoints_xyc.shape[0]*keypoints_xyc.shape[1], -1)
                else:  # for Occ-PoseTrack21-ReID
                    negative_kps = sample.pop('negative_kps')
                    keypoints_xyc = sample.pop('keypoints_xyc')
                    all_kps = [np.expand_dims(keypoints_xyc, 0)]
                    if len(negative_kps) > 0:
                        all_kps.append(negative_kps)
                    keypoints_xyc = np.concatenate(all_kps, axis=0)
                    kps_shape = keypoints_xyc.shape
                    keypoints_xyc = keypoints_xyc.reshape(keypoints_xyc.shape[0]*keypoints_xyc.shape[1], -1)

                transf_args["keypoints"] = keypoints_xyc[:, :2].astype(float)
                transf_args["kp_vis_score"] = keypoints_xyc[:, 2]
                transf_args["kp_indices"] = list(range(len(transf_args["keypoints"])))
            else:
                # if keypoints enabled but no keypoints provided as input, create empty keypoints as it is required by
                # the transforms.
                n_kp = 17
                transf_args["keypoints"] = np.zeros((n_kp, 2))
                transf_args["kp_vis_score"] = np.zeros((n_kp))
                transf_args["kp_indices"] = list(range(n_kp))
        else:
            if "keypoints_xyc" in sample:
                sample.pop("keypoints_xyc")
            if "negative_kps" in sample:
                sample.pop("negative_kps")
            if "kp_path" in sample:
                sample.pop("kp_path")
        if masks_enabled:
            if load_masks and 'masks_path' in sample:
                transf_args["masks"] = [read_masks(sample['masks_path'])]  # FIXME use BasicTransform.add_targets() to create prompt masks target
            elif not load_masks:
                # hack for BoT and PCB masks that are generated in transform().
                # FIXME BoT and PCB masks should not be generated here, but later in BPBreID model with a config
                transf_args["masks"] = [np.ones((1, 2, 2))]
            else:
                pass

        # main transforms of images, masks and keypoints
        result = transforms(**transf_args)

        img_size = (result["image"].shape[2], result["image"].shape[1])
        if kp_enabled:
            spatial_feature_size = (
            spatial_feature_shape[1], spatial_feature_shape[0])  # = (W, H)
            # reformat transformed keypoints
            keypoints = result.pop("keypoints")
            kp_vis_score = result.pop("kp_vis_score")
            kp_indices = result.pop("kp_indices")

            num_skeletons = math.ceil((np.array(kp_indices).max()+1) / 17) if len(kp_indices) > 0 else 1
            keypoints_xyc = np.zeros((num_skeletons * 17, 3))
            if len(keypoints) > 0:
                keypoints_xyc[kp_indices] = np.concatenate((np.array(keypoints), np.expand_dims(np.array(kp_vis_score), axis=1)), axis=1)
            keypoints_xyc = keypoints_xyc.reshape((num_skeletons, 17, 3)).astype(float)
            target_keypoints_xyc = keypoints_xyc[0]
            negative_keypoints_xyc = keypoints_xyc[1:]

            # build masks from keypoints
            if pose_encoding_strategy == "spatialize_part_tokens":
                prompt_mask_size = spatial_feature_size
            else:
                prompt_mask_size = img_size

            if keypoints_to_target_masks.mode != 'none':
                target_masks = keypoints_to_target_masks(target_keypoints_xyc, img_size, spatial_feature_size)
                target_masks = target_preprocess[0].apply_to_mask(torch.from_numpy(target_masks))
                sample["target_masks"] = target_preprocess[1].apply_to_mask(target_masks)  # Add background mask
            if masks_prompting and keypoints_to_prompt_masks.mode != 'none':
                prompt_masks = keypoints_to_prompt_masks(target_keypoints_xyc, img_size, prompt_mask_size)
                prompt_masks = prompt_preprocess[0].apply_to_mask(torch.from_numpy(prompt_masks))
                if use_negative_keypoints:
                    if len(negative_keypoints_xyc) != 0:
                        negative_masks = []
                        negative_skeletons = []
                        for neg_kps in negative_keypoints_xyc:
                            neg_mask = keypoints_to_prompt_masks(neg_kps, img_size, prompt_mask_size)  # TODO loop over skeletons and merge: needed for joints
                            negative_masks.append(neg_mask)
                            neg_kps = prompt_preprocess[0].apply_to_keypoints_xyc(neg_kps)
                            negative_skeletons.append(neg_kps)
                        negative_keypoints_xyc = np.stack(negative_skeletons)
                        negative_mask = torch.from_numpy(np.concatenate(negative_masks)).max(dim=0, keepdim=True)[0]
                    else:
                        negative_mask = torch.zeros((1, prompt_masks.shape[1], prompt_masks.shape[2]))
                        negative_keypoints_xyc = np.zeros((0, 17, 4))
                    prompt_masks = torch.cat((negative_mask, prompt_masks), dim=0)

                    # Padding with empty skeletons: dirty fix to have fixed size "negative_keypoints_xyc" array
                    max_skeletons = 5
                    if negative_keypoints_xyc.shape[0] > max_skeletons:
                        sample["negative_keypoints_xyc"] = negative_keypoints_xyc[:max_skeletons]
                    else:
                        sample["negative_keypoints_xyc"] = np.concatenate(
                            (
                            negative_keypoints_xyc, np.zeros((max_skeletons - negative_keypoints_xyc.shape[0], 17, 4))))

                # apply AddBackgroundMask transform
                sample["prompt_masks"] = prompt_preprocess[1].apply_to_mask(prompt_masks)
                sample["keypoints_xyc"] = prompt_preprocess[0].apply_to_keypoints_xyc(target_keypoints_xyc)

        if masks_enabled and 'masks' in result:
            masks = result.pop("masks")
            if not kp_enabled or \
                    (kp_enabled and keypoints_to_target_masks.mode == 'none'):
                sample["target_masks"] = masks[0]
            if masks_prompting:
                if not kp_enabled or \
                        (kp_enabled and keypoints_to_prompt_masks.mode == 'none'):
                    # sample["prompt_masks"] = masks[0]
                    sample["prompt_masks"] = torch.cat([masks[0][:1], masks[0][1:].max(axis=0)[0].unsqueeze(0)])  # merge into segmentation mask with one channel

        if all_occluders:  # needed for batch aware person copy paste to generate random occlusions
            result.pop("batch_pids")
            result.pop("all_occluders")

        # update sample with transformed data
        sample.update(result)

        return sample

    def show_summary(self):
        num_train_pids, num_train_cams = self.parse_data(self.train)
        num_query_pids, num_query_cams = self.parse_data(self.query)
        num_gallery_pids, num_gallery_cams = self.parse_data(self.gallery)

        print('=> Loaded {}'.format(self.__class__.__name__))
        print('  ----------------------------------------')
        print('  subset   | # ids | # images | # cameras')
        print('  ----------------------------------------')
        print(
            '  train    | {:5d} | {:8d} | {:9d}'.format(
                num_train_pids, len(self.train), num_train_cams
            )
        )
        print(
            '  query    | {:5d} | {:8d} | {:9d}'.format(
                num_query_pids, len(self.query), num_query_cams
            )
        )
        print(
            '  gallery  | {:5d} | {:8d} | {:9d}'.format(
                num_gallery_pids, len(self.gallery), num_gallery_cams
            )
        )
        print('  ----------------------------------------')


class VideoDataset(Dataset):
    """A base class representing VideoDataset.

    All other video datasets should subclass it.

    ``__getitem__`` returns an image given index.
    It will return ``imgs``, ``pid`` and ``camid``
    where ``imgs`` has shape (seq_len, channel, height, width). As a result,
    data in each batch has shape (batch_size, seq_len, channel, height, width).
    """

    def __init__(
        self,
        train,
        query,
        gallery,
        seq_len=15,
        sample_method='evenly',
        **kwargs
    ):
        super(VideoDataset, self).__init__(train, query, gallery, **kwargs)
        self.seq_len = seq_len
        self.sample_method = sample_method

        if self.transform is None:
            raise RuntimeError('transform must not be None')

    def getitem(self, index, mode):
        img_paths, pid, camid = self.data(mode)[index]  # FIXME new format
        num_imgs = len(img_paths)

        if self.sample_method == 'random':
            # Randomly samples seq_len images from a tracklet of length num_imgs,
            # if num_imgs is smaller than seq_len, then replicates images
            indices = np.arange(num_imgs)
            replace = False if num_imgs >= self.seq_len else True
            indices = np.random.choice(
                indices, size=self.seq_len, replace=replace
            )
            # sort indices to keep temporal order (comment it to be order-agnostic)
            indices = np.sort(indices)

        elif self.sample_method == 'evenly':
            # Evenly samples seq_len images from a tracklet
            if num_imgs >= self.seq_len:
                num_imgs -= num_imgs % self.seq_len
                indices = np.arange(0, num_imgs, num_imgs / self.seq_len)
            else:
                # if num_imgs is smaller than seq_len, simply replicate the last image
                # until the seq_len requirement is satisfied
                indices = np.arange(0, num_imgs)
                num_pads = self.seq_len - num_imgs
                indices = np.concatenate(
                    [
                        indices,
                        np.ones(num_pads).astype(np.int32) * (num_imgs-1)
                    ]
                )
            assert len(indices) == self.seq_len

        elif self.sample_method == 'all':
            # Samples all images in a tracklet. batch_size must be set to 1
            indices = np.arange(num_imgs)

        else:
            raise ValueError(
                'Unknown sample method: {}'.format(self.sample_method)
            )

        imgs = []
        for index in indices:
            img_path = img_paths[int(index)]
            img = read_image(img_path)
            if self.transform is not None:
                img = self.transform(img)
            img = img.unsqueeze(0) # img must be torch.Tensor
            imgs.append(img)
        imgs = torch.cat(imgs, dim=0)

        return imgs, pid, camid

    def show_summary(self):
        num_train_pids, num_train_cams = self.parse_data(self.train)
        num_query_pids, num_query_cams = self.parse_data(self.query)
        num_gallery_pids, num_gallery_cams = self.parse_data(self.gallery)

        print('=> Loaded {}'.format(self.__class__.__name__))
        print('  -------------------------------------------')
        print('  subset   | # ids | # tracklets | # cameras')
        print('  -------------------------------------------')
        print(
            '  train    | {:5d} | {:11d} | {:9d}'.format(
                num_train_pids, len(self.train), num_train_cams
            )
        )
        print(
            '  query    | {:5d} | {:11d} | {:9d}'.format(
                num_query_pids, len(self.query), num_query_cams
            )
        )
        print(
            '  gallery  | {:5d} | {:11d} | {:9d}'.format(
                num_gallery_pids, len(self.gallery), num_gallery_cams
            )
        )
        print('  -------------------------------------------')
