from collections import OrderedDict

import numpy as np

from torchreid.data.masks_transforms.mask_transform import MaskGroupingTransform

COCO_KEYPOINTS = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]

POSETRACK21_KEYPOINTS = [
    'nose',
    'head_bottom',
    'head_top',
    'left_ear',
    'right_ear',
    'left_shoulder',
    'right_shoulder',
    'left_elbow',
    'right_elbow',
    'left_wrist',
    'right_wrist',
    'left_hip',
    'right_hip',
    'left_knee',
    'right_knee',
    'left_ankle',
    'right_ankle',
]

COCO_KEYPOINTS_MAP = {k: i for i, k in enumerate(COCO_KEYPOINTS)}

COCO_JOINTS = [
    'head',
    'torso',
    'right_upperarm',
    'left_upperarm',
    'right_forearm',
    'left_forearm',
    'right_femur',
    'left_femur',
    'right_tibia',
    'left_tibia',
]

COCO_JOINTS_MAP = {k: i for i, k in enumerate(COCO_JOINTS)}


class CocoToEightBodyMasks(MaskGroupingTransform):
    parts_grouping = OrderedDict({
        "head": ["nose", "left_eye", "right_eye", "left_ear", "right_ear"],
        "left_arm": ["left_shoulder", "left_elbow", "left_wrist"],
        "right_arm": ["right_shoulder", "right_elbow", "right_wrist"],
        "torso": ["left_shoulder", "right_shoulder", "left_hip", "right_hip"],
        "left_leg": ["left_knee", "left_ankle"],
        "right_leg": ["right_knee", "right_ankle"],
        "left_feet": ["left_ankle"],
        "right_feet": ["right_ankle"],
    })

    def __init__(self):
        super().__init__(self.parts_grouping, COCO_KEYPOINTS_MAP)


class CocoToSixBodyMasks(MaskGroupingTransform):
    parts_grouping = OrderedDict({
        "head": ["nose", "left_eye", "right_eye", "left_ear", "right_ear"],
        "left_arm": ["left_elbow", "left_wrist"],
        "right_arm": ["right_elbow", "right_wrist"],
        "left_leg": ["left_knee", "left_ankle"],
        "right_leg": ["right_knee", "right_ankle"],
        "torso": ["left_shoulder", "right_shoulder", "left_hip", "right_hip"],
    })

    def __init__(self):
        super().__init__(self.parts_grouping, COCO_KEYPOINTS_MAP)


class CocoToFourBodyMasks(MaskGroupingTransform):
    parts_grouping = OrderedDict({
        "head": ["nose", "left_eye", "right_eye", "left_ear", "right_ear"],
        "arms": ["left_elbow", "left_wrist", "right_elbow", "right_wrist"],
        "legs": ["left_knee", "left_ankle", "right_knee", "right_ankle"],
        "torso": ["left_shoulder", "right_shoulder", "left_hip", "right_hip"],
    })

    def __init__(self):
        super().__init__(self.parts_grouping, COCO_KEYPOINTS_MAP)


class CocoToThreeBodyMasks(MaskGroupingTransform):
    parts_grouping = OrderedDict({
        "head": ["nose", "left_eye", "right_eye", "left_ear", "right_ear"],
        "legs": ["left_knee", "left_ankle", "right_knee", "right_ankle"],
        "torso": ["left_shoulder", "right_shoulder", "left_hip", "right_hip", "left_elbow", "left_wrist", "right_elbow", "right_wrist"],
    })

    def __init__(self):
        super().__init__(self.parts_grouping, COCO_KEYPOINTS_MAP)


class CocoToTwoBodyMasks(MaskGroupingTransform):
    parts_grouping = OrderedDict({
        "legs": ["left_knee", "left_ankle", "right_knee", "right_ankle"],
        "torso": ["nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_shoulder", "right_shoulder", "left_hip", "right_hip", "left_elbow", "left_wrist", "right_elbow", "right_wrist"],
    })

    def __init__(self):
        super().__init__(self.parts_grouping, COCO_KEYPOINTS_MAP)


class CocoToOneBodyMasks(MaskGroupingTransform):
    parts_grouping = OrderedDict({
        "body": ["left_knee", "left_ankle", "right_knee", "right_ankle", "nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_shoulder", "right_shoulder", "left_hip", "right_hip", "left_elbow", "left_wrist", "right_elbow", "right_wrist"],
    })

    def __init__(self):
        super().__init__(self.parts_grouping, COCO_KEYPOINTS_MAP)


class CocoJointsToSixBodyMasks(MaskGroupingTransform):
    # dict order matter since we want pixels to be assigned to the part with the lowest index when two parts overlap.
    # for instance, if an arm and torso overlap, pixels should be assigned to the arm. If the arm was behind the torso,
    # we make the assumption that the arm would not be visible and therefore no mask is generated for it.
    parts_grouping = OrderedDict({
        "head": ["head"],
        "left_arm": ["left_upperarm", "left_forearm"],
        "right_arm": ["right_upperarm", "right_forearm"],
        "left_leg": ["left_femur", "left_tibia"],
        "right_leg": ["right_femur", "right_tibia"],
        "torso": ["torso"],
    })

    def coco_joints_to_body_part_visibility_scores(self, coco_joints_visibility_scores):
        visibility_scores = []
        for i, part in enumerate(self.parts_names):
            visibility_scores.append(coco_joints_visibility_scores[[self.parts_map[k] for k in self.parts_grouping[part]]].mean())
        return np.array(visibility_scores)

    def __init__(self):
        super().__init__(self.parts_grouping, COCO_JOINTS_MAP)


class CocoJointsToFourBodyMasks(MaskGroupingTransform):
    # dict order matter since we want pixels to be assigned to the part with the lowest index when two parts overlap.
    # for instance, if an arm and torso overlap, pixels should be assigned to the arm. If the arm was behind the torso,
    # we make the assumption that the arm would not be visible and therefore no mask is generated for it.
    parts_grouping = OrderedDict({
        "head": ["head"],
        "arms": ["left_upperarm", "left_forearm", "right_upperarm", "right_forearm"],
        "legs": ["left_femur", "left_tibia", "right_femur", "right_tibia"],
        "torso": ["torso"],
    })

    def coco_joints_to_body_part_visibility_scores(self, coco_joints_visibility_scores):
        visibility_scores = []
        for i, part in enumerate(self.parts_names):
            visibility_scores.append(coco_joints_visibility_scores[[self.parts_map[k] for k in self.parts_grouping[part]]].mean())
        return np.array(visibility_scores)

    def __init__(self):
        super().__init__(self.parts_grouping, COCO_JOINTS_MAP)


class CocoJointsToThreeBodyMasks(MaskGroupingTransform):
    # dict order matter since we want pixels to be assigned to the part with the lowest index when two parts overlap.
    # for instance, if an arm and torso overlap, pixels should be assigned to the arm. If the arm was behind the torso,
    # we make the assumption that the arm would not be visible and therefore no mask is generated for it.
    parts_grouping = OrderedDict({
        "head": ["head"],
        "legs": ["left_femur", "left_tibia", "right_femur", "right_tibia"],
        "torso": ["torso", "left_upperarm", "left_forearm", "right_upperarm", "right_forearm"],
    })

    def coco_joints_to_body_part_visibility_scores(self, coco_joints_visibility_scores):
        visibility_scores = []
        for i, part in enumerate(self.parts_names):
            visibility_scores.append(coco_joints_visibility_scores[[self.parts_map[k] for k in self.parts_grouping[part]]].mean())
        return np.array(visibility_scores)

    def __init__(self):
        super().__init__(self.parts_grouping, COCO_JOINTS_MAP)


class CocoJointsToTwoBodyMasks(MaskGroupingTransform):
    # dict order matter since we want pixels to be assigned to the part with the lowest index when two parts overlap.
    # for instance, if an arm and torso overlap, pixels should be assigned to the arm. If the arm was behind the torso,
    # we make the assumption that the arm would not be visible and therefore no mask is generated for it.
    parts_grouping = OrderedDict({
        "legs": ["left_femur", "left_tibia", "right_femur", "right_tibia"],
        "torso": ["head", "torso", "left_upperarm", "left_forearm", "right_upperarm", "right_forearm"],
    })

    def coco_joints_to_body_part_visibility_scores(self, coco_joints_visibility_scores):
        visibility_scores = []
        for i, part in enumerate(self.parts_names):
            visibility_scores.append(coco_joints_visibility_scores[[self.parts_map[k] for k in self.parts_grouping[part]]].mean())
        return np.array(visibility_scores)

    def __init__(self):
        super().__init__(self.parts_grouping, COCO_JOINTS_MAP)


class CocoJointsToOneBodyMasks(MaskGroupingTransform):
    # dict order matter since we want pixels to be assigned to the part with the lowest index when two parts overlap.
    # for instance, if an arm and torso overlap, pixels should be assigned to the arm. If the arm was behind the torso,
    # we make the assumption that the arm would not be visible and therefore no mask is generated for it.
    parts_grouping = OrderedDict({
        "body": ["head", "torso", "left_upperarm", "left_forearm", "right_upperarm", "right_forearm", "left_femur", "left_tibia", "right_femur", "right_tibia"],
    })

    def coco_joints_to_body_part_visibility_scores(self, coco_joints_visibility_scores):
        visibility_scores = []
        for i, part in enumerate(self.parts_names):
            visibility_scores.append(coco_joints_visibility_scores[[self.parts_map[k] for k in self.parts_grouping[part]]].mean())
        return np.array(visibility_scores)

    def __init__(self):
        super().__init__(self.parts_grouping, COCO_JOINTS_MAP)
