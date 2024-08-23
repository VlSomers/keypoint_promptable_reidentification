from __future__ import absolute_import, division, print_function

import json
import os
import os.path as osp
from os import path as osp

import numpy as np
import pandas as pd
import tqdm
from ..dataset import ImageDataset


# Source: https://github.com/phil-bergmann/tracking_wo_bnw

# combine all MOTchallenge datasets into one:
#   average over all datasets, not over all queries: cannot average on unique dataset?
#   do not average with other targets: market etc: cannot average on target sets
#
# TODO put all queries in gallery
# average all scores to obtain global score
# cannot train on both market and MOTchallenge

def read_json(path):
    with open(path) as json_file:
        data = json.load(json_file)
    return data


def relabel_ids(df):
    df.rename(columns={'pid': 'pid_old'}, inplace=True)

    # Relabel Ids from 0 to N-1
    ids_df = df[['pid_old']].drop_duplicates()
    ids_df['pid'] = np.arange(ids_df.shape[0])
    df = df.merge(ids_df)
    return df


def to_dict_list(df):
    return df.to_dict('records')


def random_sampling_per_pid(df, ratio=1.0):
    def uniform_tracklet_sampling(_df):
        x = list(_df.unique())
        assert len(x) == len(_df)
        return list(np.random.choice(x, size=int(np.rint(len(x) * ratio)), replace=False))

    per_pid = df.groupby('pid')['index'].agg(uniform_tracklet_sampling)
    return per_pid.explode()


def split_by_ids(df, ratio):
    np.random.seed(0)
    ids = df['pid'].unique()
    # Generate a uniform random sample from np.arange(len(ids)):
    first_split_ids = np.random.choice(ids, int(np.rint(len(ids)*ratio)), replace=False)
    first_split = df[df['pid'].isin(first_split_ids)].copy()
    second_split = df[-df['pid'].isin(first_split_ids)].copy()
    assert not any(first_split['pid'].isin(second_split['pid']))
    assert len(first_split) + len(second_split) == len(df)
    first_split.reset_index(inplace=True)
    second_split.reset_index(inplace=True)
    first_split['index'] = first_split.index.values
    second_split['index'] = second_split.index.values
    return first_split, second_split


class MOTChallenge(ImageDataset):
    dataset_dir = 'MOTChallenge/MOT17'
    eval_metric = 'motchallenge'
    sequences = ['MOT17-02', 'MOT17-04', 'MOT17-05', 'MOT17-09', 'MOT17-10', 'MOT17-11', 'MOT17-13']

    masks_dirs = {
        # dir_name: (masks_stack_size, contains_background_mask)
    }

    @staticmethod
    def get_masks_config(masks_dir):
        if masks_dir not in MOTChallenge.masks_dirs:
            return None
        else:
            return masks_dir[masks_dir]

    def __init__(self, config, seq_name=None, masks_dir=None, **kwargs):
        self.masks_dir = masks_dir
        if self.masks_dir in self.masks_dirs:
            self.masks_parts_numbers, self.has_background, self.masks_suffix = self.masks_dirs[self.masks_dir]
        else:
            self.masks_parts_numbers, self.has_background, self.masks_suffix = None, None, None
        self.seq_names = [seq_name] if seq_name is not None else 0
        self.min_vis = config.data.mot.min_vis
        self.min_h = config.data.mot.min_h
        self.min_w = config.data.mot.min_w
        self.min_samples_per_id = config.data.mot.min_samples_per_id
        self.max_samples_per_id = config.data.mot.max_samples_per_id
        self.train_ratio = config.data.mot.train_ratio
        self.ratio_query_per_id = config.data.mot.ratio_query_per_id
        self.ratio_gallery_per_id = config.data.mot.ratio_gallery_per_id

        assert self.max_samples_per_id >= self.min_samples_per_id

        root_dir = osp.join(osp.abspath(osp.expanduser(kwargs['root'])), self.dataset_dir)
        print(f"Preparing MOTSeqDataset dataset {seq_name} from {root_dir}.")

        df = self.get_dataframe(root_dir)

        print("MOTChallenge {} size={} and #ids={}".format(seq_name, len(df), len(df['pid'].unique())))

        # df = self.filter_reid_samples(df,
        #                          self.min_vis,
        #                          min_h=self.min_h,
        #                          min_w=self.min_w,
        #                          min_samples=self.min_samples_per_id,
        #                          max_samples_per_id=self.max_samples_per_id)

        # random single-shot query/gallery
        train_df, test_df = split_by_ids(df, self.train_ratio)

        train_df = self.filter_reid_samples(train_df,
                                      self.min_vis,
                                      min_h=self.min_h,
                                      min_w=self.min_w,
                                      min_samples=self.min_samples_per_id,
                                      max_samples_per_id=self.max_samples_per_id)

        test_df = self.filter_reid_samples(test_df,
                                      min_samples=self.min_samples_per_id,
                                      max_samples_per_id=self.max_samples_per_id)

        train_df = relabel_ids(train_df)
        test_df = relabel_ids(test_df)
        query_df, gallery_df = self.split_query_gallery(test_df)

        train_df['camid'] = 0
        query_df['camid'] = 1
        gallery_df['camid'] = 2

        train = to_dict_list(train_df)
        query = to_dict_list(query_df)
        gallery = to_dict_list(gallery_df)

        print("MOTChallenge {} train size = {}".format(seq_name, len(train)))
        print("MOTChallenge {} query size = {}".format(seq_name, len(query)))
        print("MOTChallenge {} gallery size = {}".format(seq_name, len(gallery)))

        super(MOTChallenge, self).__init__(train, query, gallery, **kwargs)

    def filter_reid_samples(self, df, min_vis=0, min_h=0, min_w=0, min_samples=0, max_samples_per_id=10000):
        # iscrowd: 1 if object must be ignored, else 0
        # vis: bbox confidence
        # Filter by size and occlusion

        keep = (df['iscrowd'] == 0)
        clean_df = df[keep].copy()
        print("MOTChallenge {} removed because iscrowd = {}".format(self.seq_names, len(df) - len(clean_df)))

        if min_vis == 1.0:
            keep = (clean_df['visibility'] >= min_vis) & (clean_df['iscrowd'] == 0)
        else:
            keep = (clean_df['visibility'] > min_vis) & (clean_df['iscrowd'] == 0)
        clean_df_0 = clean_df[keep].copy()  # TODO
        print("MOTChallenge {} removed because not visible (min_vis={}) = {}".format(self.seq_names, min_vis, len(clean_df) - len(clean_df_0)))

        keep = (clean_df_0['height'] >= min_h) & (clean_df_0['width'] >= min_w)
        clean_df_1 = clean_df_0[keep].copy()  # TODO
        print("MOTChallenge {} removed because too small samples (h<{} or w<{}) = {}".format(self.seq_names, min_h, min_w, len(clean_df_0) - len(clean_df_1)))

        def uniform_tracklet_sampling(_df):
            num_det = len(_df)
            if num_det > max_samples_per_id:
                # Select 'max_samples_per_id' evenly spaced indices, including first and last
                indices = np.round(np.linspace(0, num_det - 1, max_samples_per_id)).astype(int)
                assert len(indices) == max_samples_per_id
                return _df.iloc[indices]
            else:
                return _df

        clean_df_2 = clean_df_1.groupby('pid').apply(uniform_tracklet_sampling).reset_index(drop=True).copy()
        print("MOTChallenge {} removed for uniform tracklet sampling = {}".format(self.seq_names, len(clean_df_1) - len(clean_df_2)))

        # Keep only ids with at least MIN_SAMPLES appearances
        clean_df_2['samples_per_id'] = clean_df_2.groupby('pid')['height'].transform('count').values
        clean_df_3 = clean_df_2[clean_df_2['samples_per_id'] >= min_samples].copy()
        print("MOTChallenge {} removed for not enough samples per id = {}".format(self.seq_names, len(clean_df_2) - len(clean_df_3)))

        clean_df_3.reset_index(inplace=True)
        clean_df_3['index'] = clean_df_3.index.values

        print("MOTChallenge {} filtered size = {}".format(self.seq_names, len(clean_df_3)))

        return clean_df_3

    def split_query_gallery(self, df):
        np.random.seed(0)
        query_per_id = random_sampling_per_pid(df, self.ratio_query_per_id)
        query_df = df.loc[query_per_id.values].copy()
        gallery_df = df.drop(query_per_id).copy()

        gallery_per_id = random_sampling_per_pid(gallery_df, self.ratio_gallery_per_id)
        gallery_df = gallery_df.loc[gallery_per_id.values].copy()

        return query_df, gallery_df

    def get_dataframe(self, root_dir):
        all_rows = []
        for seq_name in self.seq_names:
            ann_file = os.path.join(root_dir, 'anns', f'{seq_name}.json')
            img_dir = os.path.join(root_dir, 'imgs')
            # Create a Pandas DataFrame out of json annotations file
            anns = read_json(ann_file)
            # Build DF from anns
            rows1 = []
            for ann in tqdm.tqdm(anns['annotations']):
                box_path = osp.join(img_dir, str(seq_name), str(ann['ped_id']), ann['filename'])
                masks_path = self.infer_masks_path(box_path, self.masks_dir, self.masks_suffix)
                row = {'img_path': box_path,
                       'masks_path': masks_path,
                       'pid': int(ann['ped_id']),
                       'height': int(ann['bbox'][-1]),
                       'width': int(ann['bbox'][-2]),
                       'iscrowd': int(ann['iscrowd']),
                       'visibility': float(ann['visibility']),
                       'seq_name': seq_name,
                       'frame_n': int(ann['frame_n'])}
                rows1.append(row)
            rows = rows1
            all_rows.extend(rows)
        return pd.DataFrame(all_rows)


def get_sequence_class(seq_name):
    dataset_class = MOTChallenge

    class MOTSeqDatasetWrapper(dataset_class):
        def __init__(self, **kwargs):
            super(MOTSeqDatasetWrapper, self).__init__(seq_name=seq_name, **kwargs)

    MOTSeqDatasetWrapper.__name__ = seq_name

    return MOTSeqDatasetWrapper
