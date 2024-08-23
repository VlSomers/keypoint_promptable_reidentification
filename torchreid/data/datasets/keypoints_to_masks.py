import cv2
import numpy as np
from collections import OrderedDict
from torchreid.utils.imagetools import gkern

########################################
#        COCO skeleton structure       #
########################################

joints_dict = OrderedDict()
joints_dict['head'] = ['nose', 'head_bottom', 'head_top', 'left_ear', 'right_ear']
joints_dict['torso'] = ['left_shoulder', 'right_shoulder', 'left_hip', 'right_hip', 'head_bottom']
# arms
joints_dict['right_upperarm'] = ['right_shoulder', 'right_elbow']
joints_dict['left_upperarm'] = ['left_shoulder', 'left_elbow']
joints_dict['right_forearm'] = ['right_elbow', 'right_wrist']
joints_dict['left_forearm'] = ['left_elbow', 'left_wrist']
# legs
joints_dict['right_femur'] = ['right_hip', 'right_knee']
joints_dict['left_femur'] = ['left_hip', 'left_knee']
joints_dict['right_tibia'] = ['right_knee', 'right_ankle']
joints_dict['left_tibia'] = ['left_knee', 'left_ankle']

joints_radius = {
    'head': 3,
    'torso': 3,
    'right_upperarm': 2,
    'left_upperarm': 2,
    'right_forearm': 2,
    'left_forearm': 2,
    'right_femur': 3,
    'left_femur': 3,
    'right_tibia': 2,
    'left_tibia': 2,
}

coco_keypoints = ['nose', 'head_bottom', 'head_top', 'left_ear', 'right_ear', 'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist', 'left_hip', 'right_hip', 'left_knee', 'right_knee', 'left_ankle', 'right_ankle']

keypoints_dict = {
    'nose': 0,
    'head_bottom': 1,
    'head_top': 2,
    'left_ear': 3,
    'right_ear': 4,
    'left_shoulder': 5,
    'right_shoulder': 6,
    'left_elbow': 7,
    'right_elbow': 8,
    'left_wrist': 9,
    'right_wrist': 10,
    'left_hip': 11,
    'right_hip': 12,
    'left_knee': 13,
    'right_knee': 14,
    'left_ankle': 15,
    'right_ankle': 16,
}

# ########################################
# #    Keypoints grouping strategies     #
# ########################################
#
# k_seventeen = {
#     'nose': ['nose'],
#     'head_bottom': ['head_bottom'],
#     'head_top': ['head_top'],
#     'left_ear': ['left_ear'],
#     'right_ear': ['right_ear'],
#     'left_shoulder': ['left_shoulder'],
#     'right_shoulder': ['right_shoulder'],
#     'left_elbow': ['left_elbow'],
#     'right_elbow': ['right_elbow'],
#     'left_wrist': ['left_wrist'],
#     'right_wrist': ['right_wrist'],
#     'left_hip': ['left_hip'],
#     'right_hip': ['right_hip'],
#     'left_knee': ['left_knee'],
#     'right_knee': ['right_knee'],
#     'left_ankle': ['left_ankle'],
#     'right_ankle': ['right_ankle'],
# }
#
# k_eleven = {
#     'head_top': ['head_top'],
#     'head_middle': ['nose', 'left_ear', 'right_ear'],
#     'head_bottom': ['head_bottom'],
#     'upper_torso': ['left_shoulder', 'right_shoulder', 'left_hip', 'right_hip'],
#     'lower_torso': ['left_hip', 'right_hip'],
#     'left_arm': ['left_elbow', 'left_wrist'],
#     'right_arm': ['right_elbow', 'right_wrist'],
#     'left_legs': ['left_knee'],
#     'right_legs': ['right_knee'],
#     'left_foot': ['left_ankle'],
#     'right_foot': ['right_ankle'],
# }
#
# k_nine = {
#     'head': ['nose', 'head_bottom', 'head_top', 'left_ear', 'right_ear'],
#     'upper_torso': ['left_shoulder', 'right_shoulder', 'left_hip', 'right_hip'],
#     'lower_torso': ['left_hip', 'right_hip'],
#     'left_arm': ['left_elbow', 'left_wrist'],
#     'right_arm': ['right_elbow', 'right_wrist'],
#     'left_legs': ['left_knee'],
#     'right_legs': ['right_knee'],
#     'left_foot': ['left_ankle'],
#     'right_foot': ['right_ankle'],
# }
#
# k_eight = {
#     'head': ['nose', 'head_bottom', 'head_top', 'left_ear', 'right_ear'],
#     'torso': ['left_shoulder', 'right_shoulder', 'left_hip', 'right_hip'],
#     'left_arm': ['left_elbow', 'left_wrist'],
#     'right_arm': ['right_elbow', 'right_wrist'],
#     'left_legs': ['left_knee'],
#     'right_legs': ['right_knee'],
#     'left_foot': ['left_ankle'],
#     'right_foot': ['right_ankle'],
# }
#
# k_five = {
#     'head': ['nose', 'head_bottom', 'head_top', 'left_ear', 'right_ear'],
#     'torso': ['left_shoulder', 'right_shoulder', 'left_hip', 'right_hip'],
#     'arms': ['left_elbow', 'right_elbow', 'left_wrist', 'right_wrist'],
#     'legs': ['left_knee', 'right_knee'],
#     'feet': ['left_ankle', 'right_ankle'],
# }
#
# k_four = {
#     'head': ['nose', 'head_bottom', 'head_top', 'left_ear', 'right_ear'],
#     'torso': ['left_shoulder', 'right_shoulder', 'left_hip', 'right_hip'],
#     'arms': ['left_elbow', 'right_elbow', 'left_wrist', 'right_wrist'],
#     'legs': ['left_knee', 'right_knee', 'left_ankle', 'right_ankle'],
# }
#
# k_three = {
#     'head': ['nose', 'head_bottom', 'head_top', 'left_ear', 'right_ear'],
#     'torso': ['left_shoulder', 'right_shoulder', 'left_hip', 'right_hip', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist'],
#     'legs': ['left_knee', 'right_knee', 'left_ankle', 'right_ankle'],
# }
#
# k_two = {
#     'torso': ['nose', 'head_bottom', 'head_top', 'left_ear', 'right_ear', 'left_shoulder', 'right_shoulder', 'left_hip', 'right_hip', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist'],
#     'legs': ['left_knee', 'right_knee', 'left_ankle', 'right_ankle'],
# }
#
# k_one = {
#     'body': keypoints_dict.keys()
# }
#
# ########################################
# #     Joints grouping strategies       #
# ########################################
#
# j_six = {
#     "head": ["nose", "left_eye", "right_eye", "left_ear", "right_ear"],
#     "torso": ["left_shoulder", "right_shoulder", "left_hip", "right_hip"],
#     "left_arm": ["left_elbow", "left_wrist"],
#     "right_arm": ["right_elbow", "right_wrist"],
#     "left_leg": ["left_knee", "left_ankle"],
#     "right_leg": ["right_knee", "right_ankle"],
# }
#
# ########################################
# #       Grouping strategies dict       #
# ########################################
#
# keypoints_grouping_strats = {
#     'k_seventeen': k_seventeen,
#     'k_eleven': k_eleven,
#     'k_nine': k_nine,
#     'k_eight': k_eight,
#     'k_five': k_five,
#     'k_four': k_four,
#     'k_three': k_three,
#     'k_two': k_two,
#     'k_one': k_one,
# }
# joints_grouping_strats = {
#     'j_six': j_six,
# }


def kp_img_to_kp_bbox(kp_xyc_img, bbox_ltwh):
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

def rescale_keypoints(rf_keypoints, size, new_size):
    """
    Rescale keypoints to new size.
    Args:
        rf_keypoints (np.ndarray): keypoints in relative coordinates, shape (K, 2)
        size (tuple): original size, (w, h)
        new_size (tuple): new size, (w, h)
    Returns:
        rf_keypoints (np.ndarray): rescaled keypoints in relative coordinates, shape (K, 2)
    """
    w, h = size
    new_w, new_h = new_size
    rf_keypoints = rf_keypoints.copy()
    rf_keypoints[..., 0] = rf_keypoints[..., 0] * new_w / w
    rf_keypoints[..., 1] = rf_keypoints[..., 1] * new_h / h

    assert ((rf_keypoints[..., 0] >= 0) & (rf_keypoints[..., 0] <= new_w)).all()
    assert ((rf_keypoints[..., 1] >= 0) & (rf_keypoints[..., 1] <= new_h)).all()

    return rf_keypoints

parts_info_per_strat = {
    "keypoints": (len(keypoints_dict), keypoints_dict.keys()),
    "keypoints_gaussian": (len(keypoints_dict), keypoints_dict.keys()),
    "joints": (len(joints_dict), joints_dict.keys()),
    "joints_gaussian": (len(joints_dict), joints_dict.keys()),
}


class KeypointsToMasks:
    # TODO use simpleclick optimized method?
    """
    We use MaskTransform here because we need the keypoints to be passed through the "masks" parameter, then transformed
    to real masks/heatmaps, then returned as "masks" parameter to be processed downstream by other masks transforms in
    the Albumentation pipeline.
    """
    """
    Where to perform the keypoints to masks transformation?
    In BPBreID: hard to implement because need torch/numpy full implementation without for loops + need to transform
        keypoints before with crop/erase/etc
    As an albumentation transform: albumentation transform will take keypoints as input param and return keypoints as
        output param, but we need to return a mask, so that following transform (crop/erase/etc) can be applied on it.
        Not sure how to do that or if clean to do that.
    In dataset get_item: get the keypoints heatmaps before calling transforms, than pass the heatmaps as "masks" to
        transforms.
        V Easier implementation, re-use already functionnal transforms for masks.
        V can use existing switch between masks type: from disk/pcb/keypoints/...
        X (same issue with above implementations): same heavy transform will be applied to a given image at each epoch...
            -> use a large in memory cache?
        X Is it efficient to not do it with albumentation? (I guess albumentation transforms are not parallelized, so it's not a problem)
        X Keypoints will still not be transformed and not available within bpbreid (only the corresponding masked will be available).
    As a preprocessing, saving heatmaps to disk when building the dataset (in the dataset class):
        V fast training because just need to load masks from disk
        X hard to manage many different versions of these masks depending on hyperparameters
        X slow dataset building before running
        X harder to try many different strategies for building the keypoints masks
    """

    def __init__(self,
                 g_scale=11,
                 draw_joints=False,
                 mode=False,
                 keypoints_to_parts=None,
                 vis_thresh=0,
                 vis_continous=False,
                 ):
        super().__init__()
        self.g_scale = g_scale
        self.mode = mode
        self.draw_joints = draw_joints
        self.keypoints_to_parts = keypoints_to_parts
        self.gaussian = None
        self.vis_thresh = vis_thresh
        self.vis_continous = vis_continous

    def __call__(self, kp_xyc, img_size, output_size, **params):
        # img_size = (cols, rows)
        # assert (0 <= kp_xyc[:, 0]).all()
        # assert (kp_xyc[:, 0] <= params['cols']).all()
        # assert (0 <= kp_xyc[:, 1]).all()
        # assert (kp_xyc[:, 1] <= params['rows']).all()

        if self.mode == "keypoints":
            kp_xyc_r = rescale_keypoints(kp_xyc, img_size, output_size)
            result = self._compute_keypoints_heatmaps(output_size, kp_xyc_r)
        elif self.mode == "keypoints_gaussian":
            kp_xyc_r = rescale_keypoints(kp_xyc, img_size, output_size)
            result = self._compute_keypoints_gaussian_heatmaps(output_size, kp_xyc_r)
        elif self.mode == "joints":
            raise NotImplementedError
        elif self.mode == "joints_gaussian":
            result = self._compute_joints_gaussian_heatmaps(output_size, kp_xyc, img_size)
        else:
            raise NotImplementedError
        return result

    def _compute_keypoints_heatmaps(self, output_size, kp_xyc):
        w, h = output_size
        keypoints_heatmaps = np.zeros((len(kp_xyc), h, w))
        kp_ixyc = np.concatenate((np.expand_dims(np.arange(kp_xyc.shape[0]), 1), kp_xyc), axis=1)
        kp_iyxc = kp_ixyc[:, [0, 2, 1, 3]]
        if self.vis_continous:
            keypoints_heatmaps[kp_iyxc[:, 0].astype(int), kp_iyxc[:, 1].astype(int), kp_iyxc[:, 2].astype(int)] = kp_iyxc[:, 3]
        else:
            kp_iyxc = kp_iyxc[kp_iyxc[:, 3] > self.vis_thresh]
            keypoints_heatmaps[kp_iyxc[:, 0].astype(int), kp_iyxc[:, 1].astype(int), kp_iyxc[:, 2].astype(int)] = 1
        return keypoints_heatmaps

    def _compute_keypoints_gaussian_heatmaps(self, output_size, kp_xyc):
        w, h = output_size
        keypoints_gaussian_heatmaps = np.zeros((len(kp_xyc), h, w))
        for i, kp in enumerate(kp_xyc):
            # do not use invisible keypoints
            if kp[2] <= self.vis_thresh and not self.vis_continous:
                continue

            kpx, kpy = kp[:2].astype(int)
            g_radius = self.get_gaussian_kernel(output_size).shape[0] // 2

            rt, rb = min(g_radius, kpy), min(g_radius, h - 1 - kpy)
            rl, rr = min(g_radius, kpx), min(g_radius, w - 1 - kpx)

            kernel = self.get_gaussian_kernel(output_size)[g_radius - rt:g_radius + rb + 1, g_radius - rl:g_radius + rr + 1]

            if self.vis_continous:
                kernel = kernel * kp[2]

            keypoints_gaussian_heatmaps[i, kpy - rt:kpy + rb + 1, kpx - rl:kpx + rr + 1] = kernel
        return keypoints_gaussian_heatmaps

    def _compute_joints_gaussian_heatmaps(self, output_size, kp_xyc, img_size):
        W, H = (32, 64)  # radius used with cv.circle were configured for that scale
        kp_xyc = rescale_keypoints(kp_xyc, img_size, (W, H))
        joints_gaussian_heatmaps = np.zeros((len(joints_dict.keys()), H, W))
        for i, (joint, keypoints) in enumerate(joints_dict.items()):
            kp_indices = [keypoints_dict[kp] for kp in keypoints]
            joint_kp_xyc = kp_xyc[kp_indices]
            heatmap = joints_gaussian_heatmaps[i]
            for kp in joint_kp_xyc:
                if kp[2] > self.vis_thresh:
                    cv2.circle(heatmap, kp[0:2].astype(int), radius=joints_radius[joint], color=1, thickness=-1)
            if joint_kp_xyc[:, 2].max() > self.vis_thresh:
                kp_contours = np.array([kp[0:2].astype(int) for kp in joint_kp_xyc if kp[2] > 0])
                cv2.drawContours(heatmap, [kp_contours], contourIdx=-1, color=1, thickness=-1)
                contours, hierarchy = cv2.findContours(np.uint8(heatmap), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                assert len(contours) > 0, f"No contour found for joint {joint} with keypoints {keypoints}"
                main_contour = contours[0]
                convexHull = cv2.convexHull(main_contour)
                cv2.drawContours(heatmap, [convexHull], contourIdx=-1, color=1, thickness=-1)

                if self.vis_continous:
                    heatmap = heatmap * joint_kp_xyc[:, 2].mean()

                joints_gaussian_heatmaps[i] = heatmap
        joints_gaussian_heatmaps = cv2.resize(joints_gaussian_heatmaps.transpose((1, 2, 0)), dsize=output_size,
                   interpolation=cv2.INTER_LINEAR).transpose((2, 0, 1))
        return joints_gaussian_heatmaps

    def get_gaussian_kernel(self, output_size):  # FIXME not thread safe
        if self.gaussian is None:
            w, h = output_size
            g_radius = int(w / self.g_scale)
            self.gaussian = gkern(g_radius * 2 + 1)
        return self.gaussian
