# Source: https://github.com/isarandi/synthetic-occlusion/blob/master/augmentation.py
import math
import os.path
import random
import xml.etree.ElementTree
import numpy as np
import cv2
import PIL.Image
import torch
from albumentations import (
    DualTransform
)
import torch.nn.functional as F

from torchreid.data.datasets.keypoints_to_masks import rescale_keypoints
from torchreid.utils.tools import read_keypoints
from scipy.ndimage import shift


class BIPO(DualTransform):  # TODO clean
    """
    The Batch-wise Inter-Person Occlusion data augmentation (BIPO), to
    generate artificial inter-person occlusions on training images, prompts,
    and human parsing labels. When provided with a training sample,
    BIPO randomly selects a different person’s image (the occluder) from the same
    training batch, contours it with a segmentation mask derived from the human
    parsing labels, and overlays it on the main image. The human parsing label and
    keypoints prompt of the training image are then updated accordingly. Finally,
    all positive keypoints from the occluder are added to the training prompt to
    serve as additional negative points.
    """
    def __init__(self,
                 path,
                 im_shape,
                 always_apply=False,
                 p=.5,
                 n=1,
                 min_overlap=0.5,
                 max_overlap=0.8,
                 pascal_vot=False,
                 pid_sampling_from_batch=False,
                 ):
        super(BIPO, self).__init__(always_apply, p)
        self.pascal_vot = pascal_vot
        if self.pascal_vot:
            print('Loading occluders from Pascal VOC dataset...')
            self.all_occluders = load_pascal_occluders(pascal_voc_root_path=path)

        self.bbox_overlaps = []
        self.pxls_overlaps = []
        self.count = 0
        self.n = n
        self.min_overlap = min_overlap
        self.max_overlap = max_overlap
        self.im_shape = im_shape
        self.bottom_limit = 0.75
        self.up_limit = 0.55
        self.batch_sampling = pid_sampling_from_batch

    def check_range(self, dimension):
        if isinstance(dimension, float) and not 0 <= dimension < 1.0:
            raise ValueError(
                "Invalid value {}. If using floats, the value should be in the range [0.0, 1.0)".format(dimension)
            )

    def apply(self, image, occluders=(), centers=(), skeletons=(), **params):
        for occluder, center, keypoints in zip(occluders, centers, skeletons):
            bbox_overlap, pxls_overlap = paste_over(im_src=occluder, im_dst=image, center=center, keypoints=keypoints)
            self.bbox_overlaps.append(bbox_overlap)
            self.pxls_overlaps.append(pxls_overlap)
        self.count += 1
        # if self.count % 10000 == 0:
        #     bbox_overlaps = np.array(self.bbox_overlaps)
        #     pxls_overlaps = np.array(self.pxls_overlaps)
        #     print("RandomOcclusion #{}: bbox_overlap=[{:.2f},{:.2f},{:.2f}], pxls_overlap=[{:.2f},{:.2f},{:.2f}]"
        #           .format(self.count,
        #                   bbox_overlaps.min(),
        #                   bbox_overlaps.max(),
        #                   bbox_overlaps.mean(),
        #                   pxls_overlaps.min(),
        #                   pxls_overlaps.max(),
        #                   pxls_overlaps.mean()
        #                   )
        #           )
        return image

    def apply_to_mask(self, image, occluders=(), centers=(), skeletons=(), **params):
        for occluder, center in zip(occluders, centers):
            paste_over(im_src=occluder, im_dst=image, center=center, is_mask=True)
        return image

    def apply_to_keypoint(self, keypoint, occluders=(), centers=(), skeletons=(), **params):
        for occluder, center, keypoints in zip(occluders, centers, skeletons):
            y, x = keypoint[:2]
            if is_keypoint_in_shifted_mask(keypoint[:2], occluder[..., -1], center):
                keypoint[4] = 0
            # if y < 0 or y > occluder.shape[0] - 1 or x < 0 or x > occluder.shape[1] - 1 or occluder[y, x] < 0.001:
            #     keypoint[4] = 0
        return tuple(keypoint)

    def apply_to_keypoints(self, keypoints, occluders=(), centers=(), skeletons=(), **params):
        keypoints = [  # type: ignore
            self.apply_to_keypoint(list(keypoint), occluders, centers, skeletons, **params)
            for keypoint in keypoints
        ]
        last_index = keypoints[-1][-1]
        for occluder, skeleton in zip(occluders, skeletons):
            # visiblity score set explicitely to 1 for occluded keypoints, corresponding pixel will ALWAYS be from the occluder
            occluder_keypoints = [(kp[0], kp[1], 0, 0, 1, last_index+i+1) for i, kp in enumerate(skeleton)]
            keypoints += occluder_keypoints
        return keypoints

    def get_params_dependent_on_targets(self, params):
        img = params["image"]
        all_occluders = params["all_occluders"]
        count = np.random.randint(1, self.n + 1)
        width_height = np.asarray([img.shape[1], img.shape[0]])
        im_area = self.im_shape[1] * self.im_shape[0]
        occluders = []
        centers = []
        skeletons = []
        for _ in range(count):
            if self.pascal_vot:
                occluder = random.choice(self.all_occluders)
                occluder_area = occluder.shape[1] * occluder.shape[0]
                overlap = random.uniform(self.min_overlap, self.max_overlap)
                scale_factor = math.sqrt(overlap * im_area / occluder_area)
                occluder = resize_by_factor(occluder, scale_factor)
                # assert abs((occluder.shape[1] * occluder.shape[0]) / im_area - overlap) < 0.005
                center = np.random.uniform([0, 0], width_height)  # FIXME between 0.2 and 0.5 of image height
                keypoints_xyc = None
            else:
                if self.batch_sampling:
                    batch_pids = list(params["batch_pids"])
                    random.shuffle(batch_pids)
                    selected_batch_pid = next((pid for pid in batch_pids if pid in all_occluders),
                                              random.choice(list(all_occluders.keys())))
                else:
                    selected_batch_pid = random.choice(list(all_occluders.keys()))
                pid_occluders = all_occluders[selected_batch_pid]
                occluder, keypoints_xyc = random.choice(pid_occluders)
                occluder = cv2.resize(occluder, width_height, interpolation=cv2.INTER_LINEAR)
                center = np.random.uniform([0, int(img.shape[0] * self.up_limit)], np.asarray([img.shape[1], int(img.shape[0] * self.bottom_limit)]))
                keypoints_xyc = rescale_keypoints(keypoints_xyc, (occluder.shape[1], occluder.shape[0]), width_height)
                bbox_ltwh = (0, 0, width_height[0], width_height[1])
                keypoints_xyc = recenter_keypoints(keypoints_xyc, (occluder.shape[1]/2, occluder.shape[0]/2), center, bbox_ltwh)

            occluders.append(occluder)
            centers.append(center)
            skeletons.append(keypoints_xyc)

        return {
                "occluders": occluders,
                "centers": centers,
                "skeletons": skeletons,
                }

    @property
    def targets_as_params(self):
        return ["image", "batch_pids", "all_occluders"]

    def get_transform_init_args_names(self):
        return (
            "max_holes",
            "max_height",
            "max_width",
            "min_holes",
            "min_height",
            "min_width",
            "fill_value",
            "mask_fill_value",
        )


def update_visibility(keypoints, mask):
    for point in keypoints:
        # If the point is outside the mask, set visibility to 0
        y, x = round(point[1]), round(point[0])
        if y<0 or y>mask.shape[0]-1 or x<0 or x>mask.shape[1]-1 or mask[y, x] < 0.001:  # note that it's y, x because mask is a 2D array
            point[2] = 0

    return keypoints


def update_visibility_np(keypoints, mask):
    keypoints_rounded = np.round(keypoints).astype(int)

    # create valid indices mask
    valid_rows = np.logical_and(keypoints_rounded[:, 1] >= 0, keypoints_rounded[:, 1] < mask.shape[0])
    valid_cols = np.logical_and(keypoints_rounded[:, 0] >= 0, keypoints_rounded[:, 0] < mask.shape[1])
    valid_indices = np.logical_and(valid_rows, valid_cols)

    # for valid indices check if mask value is < 0.001
    valid_and_masked = np.zeros_like(valid_indices)
    valid_and_masked[valid_indices] = mask[keypoints_rounded[valid_indices, 1], keypoints_rounded[
        valid_indices, 0]] < 0.001

    # set visibility to 0 for points outside the mask or where mask value < 0.001
    keypoints[valid_and_masked, 2] = 0

    return keypoints

def recenter_keypoints(keypoints, original_center, new_center, bbox):
    # compute difference between new and old center
    dx, dy = new_center[0] - original_center[0], new_center[1] - original_center[1]

    # apply difference to all keypoints
    keypoints[:, 0] += dx
    keypoints[:, 1] += dy

    # check if keypoints are inside the bounding box
    for point in keypoints:
        x, y = point[0], point[1]
        if not (bbox[0] <= x <= bbox[2] and bbox[1] <= y <= bbox[3]):  # if outside bbox
            point[2] = 0  # set visibility to 0

    return keypoints

def load_human_occluders(samples, img_size=(128, 256), max_occ_per_id=5):
    # imgs_filenames = glob.glob(os.path.join(imgs_path, '*.jpg'))  # assuming images are in .jpg format

    pid_to_sample_list = {}
    for sample in samples:  # TODO do it per camid?
        pid = sample["pid"]
        if pid not in pid_to_sample_list:
            pid_to_sample_list[pid] = []
        pid_to_sample_list[pid].append(sample)

    final_pid_to_sample_list = {}
    for pid, pid_samples in pid_to_sample_list.items():
        pid_samples = np.array(pid_samples)
        if len(pid_samples) > max_occ_per_id:
            pid_samples = np.random.choice(pid_samples, max_occ_per_id, replace=False)
        final_pid_to_sample_list[pid] = list(pid_samples)

    # print(f"Occluders = {[os.path.splitext(os.path.basename(f))[0] for f in sampled_imgs_filenames]}")
    # TODO don't use occluder if same id as image
    # TODO : at least X keypoints still visible
    imgs_per_pid = {}
    for pid, pid_samples in final_pid_to_sample_list.items():
        for sample in pid_samples:
            # for each image, load the image
            img_filename = sample["img_path"]
            img = cv2.imread(img_filename)
            or_img_size = (img.shape[1], img.shape[0])
            img = cv2.resize(img, img_size, cv2.INTER_CUBIC)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # Load the corresponding mask
            mask_filename = sample["masks_path"]
            mask = np.load(mask_filename)
            # mask = np.transpose(mask, (1, 2, 0))

            # resize the mask to match the image size
            mask = F.interpolate(
                torch.from_numpy(mask).unsqueeze(0),
                size=img.shape[:2],
                mode='bilinear',
                align_corners=True
            ).squeeze()
            mask = mask.max(0)[0]

            mask = mask.numpy()
            _, bmask = cv2.threshold(mask, 0.2, 255, cv2.THRESH_BINARY)

            # Create a kernel - you can define the size, it depends on how much dilation you want
            kernel_size = 5  # You can change this
            kernel = np.ones((kernel_size, kernel_size), np.uint8)

            # Dilate the image
            bmask = cv2.dilate(bmask, kernel, iterations=1)

            # Find the contours
            contours, _ = cv2.findContours(bmask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Check if any contours were found
            if contours:
                # Define the initial largest area as 0
                largest_area = 0
                unique_contour = None

                # Loop through the contours
                for contour in contours:
                    # Get the contour area
                    area = cv2.contourArea(contour)

                    # Check if this contour is larger than the currently largest known contour
                    if area > largest_area:
                        largest_area = area
                        unique_contour = contour

                # Draw the contour on the mask and fill it with '1's
                mask = np.zeros_like(mask)
                mask = cv2.drawContours(mask, [unique_contour], -1, (1), thickness=cv2.FILLED)

                # Apply gaussian blur to smooth the mask's borders
                mask = cv2.GaussianBlur(mask, (9, 9), 0)
            else:
                # print("No contours found in image.")
                continue

            # Load keypoints
            if "kp_path" in sample:
                keypoints_filename = sample["kp_path"]
                bbox_ltwh = (0, 0, or_img_size[0], or_img_size[1])
                pos_and_neg_keypoints_xyc = read_keypoints(keypoints_filename, bbox_ltwh)
                keypoints_xyc = pos_and_neg_keypoints_xyc[0]
            elif "keypoints_xyc" in sample:
                keypoints_xyc = sample["keypoints_xyc"]
            else:
                raise ValueError("No keypoints found for sample")
            # rescale to img size
            keypoints_xyc = rescale_keypoints(keypoints_xyc, or_img_size, img_size)

            # img = draw_keypoints(img, keypoints_xyc, [img.shape[1], img.shape[0]], vis_thresh=0,
            #                         color=(255, 255, 0))

            # remove keypoints outside mask
            keypoints_xyc = update_visibility(keypoints_xyc, mask)
            keypoints_xyc_np = update_visibility_np(keypoints_xyc, mask)
            assert np.equal(keypoints_xyc_np, keypoints_xyc).all()
            # Concat image and mask
            img_mask = np.concatenate((np.array(img), np.expand_dims(mask, axis=2) * 255), axis=-1)

            if pid not in imgs_per_pid:
                imgs_per_pid[pid] = []
            imgs_per_pid[pid].append((img_mask, keypoints_xyc.astype(float) ))

    # return the list of images
    return imgs_per_pid


def load_pascal_occluders(
        pascal_voc_root_path,
        classes_filter=None,
):
    occluders = []
    structuring_element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8, 8))
    if classes_filter is None:
        # classes_filter = ["person", "bicycle", "boat", "bus", "car", "motorbike", "train", "chair", "dining", "table", "plant", "sofa"]
        classes_filter = ["person", "bicycle", "boat", "bus", "car", "motorbike", "train"]
        # classes_filter = ["person"]
    annotation_paths = list_filepaths(os.path.join(pascal_voc_root_path, 'Annotations'))
    for annotation_path in annotation_paths:
        xml_root = xml.etree.ElementTree.parse(annotation_path).getroot()
        is_segmented = (xml_root.find('segmented').text != '0')

        if not is_segmented:
            continue

        boxes = []
        for i_obj, obj in enumerate(xml_root.findall('object')):
            is_authorized_class = (obj.find('name').text in classes_filter)
            is_difficult = (obj.find('difficult').text != '0')
            is_truncated = (obj.find('truncated').text != '0')
            if is_authorized_class and not is_difficult and not is_truncated:
                bndbox = obj.find('bndbox')
                box = [int(bndbox.find(s).text) for s in ['xmin', 'ymin', 'xmax', 'ymax']]
                boxes.append((i_obj, box))

        if not boxes:
            continue

        im_filename = xml_root.find('filename').text
        seg_filename = im_filename.replace('jpg', 'png')

        im_path = os.path.join(pascal_voc_root_path, 'JPEGImages', im_filename)
        seg_path = os.path.join(pascal_voc_root_path, 'SegmentationObject', seg_filename)

        im = np.asarray(PIL.Image.open(im_path))
        labels = np.asarray(PIL.Image.open(seg_path))

        for i_obj, (xmin, ymin, xmax, ymax) in boxes:
            object_mask = (labels[ymin:ymax, xmin:xmax] == i_obj + 1).astype(np.uint8) * 255
            object_image = im[ymin:ymax, xmin:xmax]
            if cv2.countNonZero(object_mask) < 500:
                # Ignore small objects
                continue

            # Reduce the opacity of the mask along the border for smoother blending
            eroded = cv2.erode(object_mask, structuring_element)
            object_mask[eroded < object_mask] = 192
            object_with_mask = np.concatenate([object_image, object_mask[..., np.newaxis]], axis=-1)

            # Downscale for efficiency
            object_with_mask = resize_by_factor(object_with_mask, 0.5)
            occluders.append(object_with_mask)

    return occluders


def occlude_with_objects(im, occluders, n=1, min_overlap=0.1, max_overlap=0.6):
    """Returns an augmented version of `im`, containing some occluders from the Pascal VOC dataset."""

    result = im.copy()
    width_height = np.asarray([im.shape[1], im.shape[0]])
    im_area = im.shape[1] * im.shape[0]
    count = np.random.randint(1, n+1)

    for _ in range(count):
        occluder = random.choice(occluders)
        occluder_area = occluder.shape[1] * occluder.shape[0]
        overlap = random.uniform(min_overlap, max_overlap)
        scale_factor = math.sqrt(overlap * im_area / occluder_area)
        occluder = resize_by_factor(occluder, scale_factor)
        assert (occluder.shape[1] * occluder.shape[0]) / im_area == overlap
        center = np.random.uniform([0, 0], width_height)
        paste_over(im_src=occluder, im_dst=result, center=center)
    return result


def paste_over(im_src, im_dst, center, is_mask=False, keypoints=None):
    """Pastes `im_src` onto `im_dst` at a specified position, with alpha blending, in place.

    Locations outside the bounds of `im_dst` are handled as expected (only a part or none of
    `im_src` becomes visible).

    Args:
        im_src: The RGBA image to be pasted onto `im_dst`. Its size can be arbitrary.
        im_dst: The target image.
        alpha: A float (0.0-1.0) array of the same size as `im_src` controlling the alpha blending
            at each pixel. Large values mean more visibility for `im_src`.
        center: coordinates in `im_dst` where the center of `im_src` should be placed.
    """

    width_height_src = np.asarray([im_src.shape[1], im_src.shape[0]])
    width_height_dst = np.asarray([im_dst.shape[1], im_dst.shape[0]])

    center = np.round(center).astype(np.int32)
    raw_start_dst = center - width_height_src // 2
    raw_end_dst = raw_start_dst + width_height_src

    start_dst = np.clip(raw_start_dst, 0, width_height_dst)
    end_dst = np.clip(raw_end_dst, 0, width_height_dst)
    region_dst = im_dst[start_dst[1]:end_dst[1], start_dst[0]:end_dst[0]]

    start_src = start_dst - raw_start_dst
    end_src = width_height_src + (end_dst - raw_end_dst)
    region_src = im_src[start_src[1]:end_src[1], start_src[0]:end_src[0]]
    color_src = region_src[..., 0:3]
    alpha = region_src[..., 3:].astype(np.float32) / 255

    if is_mask:  # if this is a segmentation mask, just apply alpha erasing
        im_dst[start_dst[1]:end_dst[1], start_dst[0]:end_dst[0]] = (1 - alpha) * region_dst
    else:
        im_dst[start_dst[1]:end_dst[1], start_dst[0]:end_dst[0]] = (
            alpha * color_src + (1 - alpha) * region_dst)
        im_area = im_src.shape[1] * im_src.shape[0]
        bbox_overlap = (color_src.shape[0] * color_src.shape[1]) / im_area
        pxls_overlap = np.count_nonzero(alpha) / im_area
        # im_dst = draw_keypoints(im_dst, keypoints, [im_src.shape[1], im_src.shape[0]], vis_thresh=0, color=(191, 64, 191))
        return bbox_overlap, pxls_overlap


def resize_by_factor(im, factor):
    """Returns a copy of `im` resized by `factor`, using bilinear interp for up and area interp
    for downscaling.
    """
    new_size = tuple(np.round(np.array([im.shape[1], im.shape[0]]) * factor).astype(int))
    interp = cv2.INTER_LINEAR if factor > 1.0 else cv2.INTER_AREA
    return cv2.resize(im, new_size, fx=factor, fy=factor, interpolation=interp)


def list_filepaths(dirpath):
    names = os.listdir(dirpath)
    paths = [os.path.join(dirpath, name) for name in names]
    return sorted(filter(os.path.isfile, paths))


def is_keypoint_in_shifted_mask(keypoint, mask, center):
    # Calculate the shift values as the difference between the current center and the new one
    shift_values = np.array((center[1], center[0])) - np.array(mask.shape) // 2

    # Shift the mask
    shifted_mask = shift(mask, shift_values, order=0, mode='constant', cval=0)

    # Round keypoint values and convert them to integers
    keypoint = np.round(keypoint).astype(int)

    # Check if the keypoint is within the bounds of the shifted mask
    if (0 <= keypoint[0] < shifted_mask.shape[1] and 0 <= keypoint[1] < shifted_mask.shape[0]):
        # Return True if the keypoint lies inside the mask, False otherwise
        return shifted_mask[keypoint[1], keypoint[0]] > shifted_mask.max() / 10

    return False

