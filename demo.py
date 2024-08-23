import json
import os
import cv2
import numpy as np
import torch

from pathlib import Path
from torchreid.metrics.distance import compute_distance_matrix_using_bp_features
from torchreid.scripts.builder import build_config
from torchreid.tools.feature_extractor import KPRFeatureExtractor
from torchreid.utils.visualization.display_kpr_samples import display_kpr_reid_samples_grid, display_distance_matrix

# python3 demo.py

# OVERVIEW
# This script demonstrates how to use the KPR model to extract features from images and keypoints prompts.
# It uses images and keypoints from the "assets/demo/soccer_players" folder.
# It uses Matplotlib to display the images, prompts and model outputs.
# The figures are plotted or saved in the "assets/demo/results" folder depending on the following config:
display_mode = 'save'   # 'plot' or 'save'

# /!\ IMPORTANT NOTE: Re-identification is known to be challenging in a cross domain settings. Since cross-domain ReID
# was not part of our study, I cannot guarantee KPR will work on your data.
# Finally, we also plan to release a KPR model that is trained on several datasets at the same time, hoping to improve
# its generalization capabilities (stay tuned).

# ------------------------------------
# 1) Install instructions :
# Follow the installation instructions in the main README.md file to setup your python environment
# Download the model weights from the following link: https://drive.google.com/file/d/1Np5wu3nQa_Fl_z7Zw2kchJNC8JZVwsh5/view?usp=sharing
# Put the downloaded file under '/path/to/keypoint_promptable_reidentification/pretrained_models/kpr_occ_pt_IN_82.34_92.33_42323828.pth.tar'

# ------------------------------------
# 2) Load the configuration
# go inside 'configs/kpr/imagenet/kpr_occ_posetrack_test.yaml' and change the path in the 'load_weights' config if you saved the downloaded
# weights in a different location
# -> have a look at '/torchreid/scripts/default_config.py' for detailed comments about all available options
kpr_cfg = build_config(config_path="configs/kpr/imagenet/kpr_occ_posetrack_test.yaml")
kpr_cfg.use_gpu = torch.cuda.is_available() # already done in build_config(...), but can be overwritten here
# kpr_cfg.model.promptable_trans.disable_inference_prompting = True  # Disable prompting during inference (keypoints prompts are ignored by KPR)

# ------------------------------------
# 3) Initialize the feature extractor, which is a convenient wrapper around the KPR model that handles preprocessing
# the input images and prompts, and postprocessing the outputs.
extractor = KPRFeatureExtractor(kpr_cfg)

# ------------------------------------
# 4) Load our demo samples from the "assets/demo/soccer_players" folder. This folder contains two subfolders, each
# containing images and keypoints for a group of soccer players. The keypoints are stored in JSON files with the same
# name as the corresponding image files.
def load_kpr_samples(images_folder, keypoints_folder):
    # Get a list of all image files in the folder
    image_files = [f for f in os.listdir(images_folder) if f.endswith('.jpg')]
    # Initialize an empty list to store the samples
    samples = []
    # Iterate over the image files and construct each sample dynamically
    for img_name in image_files:
        # Construct full paths
        img_path = os.path.join(images_folder, img_name)
        json_path = os.path.join(keypoints_folder, img_name.replace('.jpg', '.json'))

        # Load the image
        img = cv2.imread(img_path)

        # Load the keypoints from the JSON file
        with open(json_path, 'r') as json_file:
            keypoints_data = json.load(json_file)

        # Initialize lists to hold keypoints
        keypoints_xyc = []
        negative_kps = []

        # Process the keypoints data
        for entry in keypoints_data:
            if entry["is_target"]:
                keypoints_xyc.append(entry["keypoints"])
            else:
                negative_kps.append(entry["keypoints"])

        assert len(keypoints_xyc) == 1, "Only one target keypoint set is supported for now."

        # Convert lists to numpy arrays
        keypoints_xyc = np.array(keypoints_xyc[0])
        negative_kps = np.array(negative_kps)

        # Create the sample dictionary
        sample = {
            "image": img,
            "keypoints_xyc": keypoints_xyc,  # the positive prompts indicating the re-identification target
            "negative_kps": negative_kps,  # the negative keypoints indicating other pedestrians
        }

        # Append the sample to the list
        samples.append(sample)
    return samples


# Folders containing the images and keypoints
base_folder = Path('assets/demo/soccer_players')
group1_folder = base_folder/'group1'
group2_folder = base_folder/'group2'

samples_grp_1 = load_kpr_samples(group1_folder/'images', group1_folder/'keypoints')
samples_grp_2 = load_kpr_samples(group2_folder/'images', group2_folder/'keypoints')
# samples_grp_1 and samples_grp_2 are lists of dictionaries, each dictionary containing three field: "image",
# "keypoints_xyc" (positive keypoints) and "negative_kps" (negative keypoints). Both "keypoints_xyc" and "negative_kps"
# are optional and can be omitted if not available. In this case, KPR will perform re-identification based on the image
# only, without using keypoints prompts.

# ------------------------------------
# 5) Display all the samples in a grid. Keypoints prompts are displayed in the image as dots, with one color per body part, while pure red dots indicates negative keypoints.
display_kpr_reid_samples_grid(samples_grp_1 + samples_grp_2, display_mode=display_mode)

# ------------------------------------
# 6) Extract features for both groups of samples. The KPRFeatureExtractor returns the updated samples list with three
# new values: the 'embeddings', 'visibility_scores', and 'parts_masks' as numpy arrays. It also returns the raw batched
# torch tensors with embeddings, visibility scores, and parts masks, for further processing if needed.
# keypoints are optional: the "keypoints_xyc" and "negative_kps" keys can be omitted from the samples if not available.
# "negative_kps" should contain an empty array if no negative keypoints are available.
samples_grp_1, embeddings_grp_1, visibility_scores_grp_1, parts_masks_grp_1 = extractor(samples_grp_1)
samples_grp_2, embeddings_grp_2, visibility_scores_grp_2, parts_masks_grp_2 = extractor(samples_grp_2)

# ------------------------------------
# 7) Call again the display function, this time with the updated samples, to visualize the part attention maps output by
# the model.
display_kpr_reid_samples_grid(samples_grp_1 + samples_grp_2, display_mode=display_mode, save_path='assets/demo/results/samples_grid.png')

# ------------------------------------
# 8) Compute the distance matrix between the first and the second group of samples. A distance close to 0 indicate a
# strong appearance similarity (samples have likely the same identity) and a distance close to 1 indicate a strong
# appearance dissimilarity (samples have likely different identities).
distance_matrix, body_parts_distmat = compute_distance_matrix_using_bp_features(embeddings_grp_1,
                                                      embeddings_grp_2,
                                                      visibility_scores_grp_1,
                                                      visibility_scores_grp_2,
                                                      use_gpu=False,
                                                      use_logger=False
                                                      )
distances = distance_matrix.cpu().detach().numpy() / 2  # The above function returns distances within the [0, 2] range

# ------------------------------------
# 9) Display the resulting distance matrix. /!\ Visually inspecting the part attention masks (and the computed distances)
# does not always reflect the true overall performance of the model. Please refer to the evaluation code using standard
# metrics such as mAP and Rank-1 for a more accurate evaluation.
display_distance_matrix(distances, samples_grp_1, samples_grp_2, display_mode=display_mode, save_path='assets/demo/results/distance_matrix.png')