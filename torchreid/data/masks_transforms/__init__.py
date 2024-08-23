from __future__ import print_function, absolute_import

from .mask_transform import *
from .pcb_transforms import *
from .pifpaf_mask_transform import *
from .coco_keypoints_transforms import *

masks_preprocess_pifpaf = {
    "full": CombinePifPafIntoFullBodyMask,
    "bs_fu": AddFullBodyMaskToBaseMasks,
    "bs_fu_bb": AddFullBodyMaskAndFullBoundingBoxToBaseMasks,
    "mu_sc": CombinePifPafIntoMultiScaleBodyMasks,
    "one": CombinePifPafIntoOneBodyMasks,
    "two_v": CombinePifPafIntoTwoBodyMasks,
    "three_v": CombinePifPafIntoThreeBodyMasks,
    "four": CombinePifPafIntoFourBodyMasks,
    "four_no": CombinePifPafIntoFourBodyMasksNoOverlap,
    "four_v": CombinePifPafIntoFourVerticalParts,
    "four_v_pif": CombinePifPafIntoFourVerticalPartsPif,
    "five_v": CombinePifPafIntoFiveVerticalParts,
    "five": CombinePifPafIntoFiveBodyMasks,
    "six": CombinePifPafIntoSixBodyMasks,
    "six_v": CombinePifPafIntoSixVerticalParts,
    "six_no": CombinePifPafIntoSixBodyMasksSum,
    "six_new": CombinePifPafIntoSixBodyMasksSimilarToEight,
    "seven_v": CombinePifPafIntoSevenVerticalBodyMasks,
    "seven_new": CombinePifPafIntoSevenBodyMasksSimilarToEight,
    "eight": CombinePifPafIntoEightBodyMasks,
    "eight_v": CombinePifPafIntoEightVerticalBodyMasks,
    "ten_ms": CombinePifPafIntoTenMSBodyMasks,
    "eleven": CombinePifPafIntoElevenBodyMasks,
    "fourteen": CombinePifPafIntoFourteenBodyMasks,
}

masks_preprocess_coco = {
    "cck8": CocoToEightBodyMasks,
    "cck6": CocoToSixBodyMasks,
    "cck4": CocoToFourBodyMasks,
    "cck3": CocoToThreeBodyMasks,
    "cck2": CocoToTwoBodyMasks,
    "cck1": CocoToOneBodyMasks,
}

masks_preprocess_coco_joints = {
    "ccj6": CocoJointsToSixBodyMasks,
    "ccj4": CocoJointsToFourBodyMasks,
    "ccj3": CocoJointsToThreeBodyMasks,
    "ccj2": CocoJointsToTwoBodyMasks,
    "ccj1": CocoJointsToOneBodyMasks,
}

masks_preprocess_fixed = {
    "id": FullMask,
    "strp_2": PCBMasks2,
    "strp_3": PCBMasks3,
    "strp_4": PCBMasks4,
    "strp_5": PCBMasks5,
    "strp_6": PCBMasks6,
    "strp_7": PCBMasks7,
    "strp_8": PCBMasks8,
}

masks_preprocess_transforms = {**masks_preprocess_pifpaf, **masks_preprocess_coco, **masks_preprocess_coco_joints}
masks_preprocess_all = {
    **masks_preprocess_pifpaf,
    **masks_preprocess_fixed,
    **masks_preprocess_coco,
    **masks_preprocess_coco_joints,
    'none': IdentityMaskTransform
}
