import cv2
import numpy as np
from collections import OrderedDict
from scipy.signal.windows import gaussian

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


def gkern(kernlen=21, std=None):
    """Returns a 2D Gaussian kernel array."""
    if std is None:
        std = kernlen / 4
    gkern1d = gaussian(kernlen, std=std).reshape(kernlen, 1)
    gkern2d = np.outer(gkern1d, gkern1d)
    return gkern2d


def build_keypoints_heatmaps(kp_xyc, w, h):
    keypoints_heatmaps = np.zeros((len(kp_xyc), h, w))
    kp_ixyc = np.concatenate((np.expand_dims(np.arange(kp_xyc.shape[0]), 1), kp_xyc.astype(int)), axis=1)
    kp_ixy = kp_ixyc[kp_ixyc[:, 3] != 0][:, :3]
    kp_iyx = kp_ixy[:, [0, 2, 1]]
    keypoints_heatmaps[kp_iyx[:, 0], kp_iyx[:, 1], kp_iyx[:, 2]] = 1
    return keypoints_heatmaps.astype(np.uint8)


def build_keypoints_gaussian_heatmaps(kp_xyc, w, h, gaussian=None):
    gaussian_heatmaps = np.zeros((len(kp_xyc), h, w))
    for i, kp in enumerate(kp_xyc):
        # do not use invisible keypoints
        if kp[2] == 0:
            continue

        kpx, kpy = kp[:2].astype(int)

        if gaussian is None:
            g_scale = 8
            g_radius = int(w / g_scale)
            gaussian = gkern(g_radius * 2 + 1)
        else:
            g_radius = gaussian.shape[0] // 2

        rt, rb = min(g_radius, kpy), min(g_radius, h - 1 - kpy)
        rl, rr = min(g_radius, kpx), min(g_radius, w - 1 - kpx)

        gaussian_heatmaps[i, kpy - rt:kpy + rb + 1, kpx - rl:kpx + rr + 1] = gaussian[
                                                                             g_radius - rt:g_radius + rb + 1,
                                                                             g_radius - rl:g_radius + rr + 1]
    return gaussian_heatmaps


def build_joints_heatmaps(kp_xyc, w, h):
    gaussian_heatmaps = np.zeros((len(joints_dict.keys()), h, w))

    for i, (joint, keypoints) in enumerate(joints_dict.items()):
        kp_indices = [keypoints_dict[kp] for kp in keypoints]
        joint_kp_xyc = kp_xyc[kp_indices]
        heatmap = gaussian_heatmaps[i]
        for kp in joint_kp_xyc:
            if kp[2] > 0:  # kp[0:2].astype(int) = (x, y) (w, h)
                cv2.circle(heatmap, kp[0:2].astype(int), radius=joints_radius[joint], color=1, thickness=-1)
        if joint_kp_xyc[:, 2].max() != 0:
            raise NotImplementedError
            # TODO draw lines
            # kp_contours = np.array([kp[0:2].astype(int) for kp in joint_kp_xyc if kp[2] > 0])
            # cv2.drawContours(heatmap, [kp_contours], contourIdx=-1, color=1, thickness=-1)
            # contours, hierarchy = cv2.findContours(np.uint8(heatmap), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            # main_contour = contours[0]
            # convexHull = cv2.convexHull(main_contour)
            # cv2.drawContours(heatmap, [convexHull], contourIdx=-1, color=1, thickness=-1)

    return gaussian_heatmaps


def build_joints_gaussian_heatmaps(kp_xyc, w, h):
    gaussian_heatmaps = np.zeros((len(joints_dict.keys()), h, w))

    for i, (joint, keypoints) in enumerate(joints_dict.items()):
        kp_indices = [keypoints_dict[kp] for kp in keypoints]
        joint_kp_xyc = kp_xyc[kp_indices]
        heatmap = gaussian_heatmaps[i]
        for kp in joint_kp_xyc:
            if kp[2] > 0:  # kp[0:2].astype(int) = (x, y) (w, h)
                cv2.circle(heatmap, kp[0:2].astype(int), radius=joints_radius[joint], color=1, thickness=-1)
        if joint_kp_xyc[:, 2].max() != 0:
            kp_contours = np.array([kp[0:2].astype(int) for kp in joint_kp_xyc if kp[2] > 0])
            cv2.drawContours(heatmap, [kp_contours], contourIdx=-1, color=1, thickness=-1)
            contours, hierarchy = cv2.findContours(np.uint8(heatmap), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            main_contour = contours[0]
            convexHull = cv2.convexHull(main_contour)
            cv2.drawContours(heatmap, [convexHull], contourIdx=-1, color=1, thickness=-1)

    return gaussian_heatmaps


def keypoints_to_body_part_visibility_scores(kp_xyc):
    visibility_scores = []
    for i, (joint, keypoints) in enumerate(joints_dict.items()):
        kp_indices = [keypoints_dict[kp] for kp in keypoints]
        joint_kp_xyc = kp_xyc[kp_indices]
        visibility_scores.append(joint_kp_xyc[:, 2].mean())

    return np.array(visibility_scores)
