import torch
import torch.nn as nn
import numpy as np

from segment_anything import sam_model_registry
from segment_anything.modeling import Sam

# TODO implement transforms and keypoints loading
# TODO rename conda env + create local
# TODO use sam submodule
# -----
# TODO try lovely-tensors?
# TODO # preprocess = cv load -> RGB -> ResizeLongestSide -> as_tensor -> permute+contiguous -> Normalize colors -> Pad
# TODO move transforms and preprocess to torchreid transforms
# TODO need points/mask etc as input of forward: change it on torchreid side
# TODO return spatial feature of size [N, D, Hf, Wf]: change masks_decoder output
# TODO need feature_dim
# TODO support multiple skeletons input with positive and negative


# TODO transform_te: Resize - Normalize - ToTensorV2 - PermuteMasksDim - CocoJointsToSixBodyMasks - AddBackgroundMask - ResizeMasks
# TODO transform_tr: Resize - PadIfNeeded - RandomCrop - Normalize - CoarseDropout - ToTensorV2 - PermuteMasksDim - CocoJointsToSixBodyMasks - AddBackgroundMask - ResizeMasks
# TODO sam img: cv load -> RGB (done) -> ResizeLongestSide (todo implement) -> as_tensor (done) -> permute+contiguous (ToTensorV2 do permute, contiguous done somewhere?) -> Normalize colors (todo) (255 range) -> Pad (todo)
# TODO sam points+labels: ResizeLongestSide -> as_tensor, however, will need crop, pad, dropout, resize, ...
# TODO sam masks: already good?


class SamReID(nn.Module):
    def __init__(
        self,
        sam_model: Sam,
    ) -> None:
        """
        Uses SAM to calculate the image embedding for an image, and then
        allow repeated, efficient mask prediction given prompts.

        Arguments:
          sam_model (Sam): The model to use for mask prediction.
        """
        super().__init__()
        self.model = sam_model
        self.feature_dim = 256
        # self.transform = ResizeLongestSide(sam_model.image_encoder.img_size)
        # self.reset_image()
        # self.return_logits = False

    def forward(self, images, masks=None, keypoints_xyc=None):
        """        point_coords: Optional[np.ndarray] = None,
        point_labels: Optional[np.ndarray] = None,
        box: Optional[np.ndarray] = None,
        mask_input: Optional[np.ndarray] = None,
        multimask_output: bool = True,
        return_logits: bool = False,"""

        # Transform the image to the form expected by the model
        # ######### MOVE ############
        # input_image = self.transform.apply_image(image)  # opencv NP image casted to RGB
        # input_image_torch = torch.as_tensor(input_image, device=self.device)  # FIXME still needed?
        # input_image_torch = input_image_torch.permute(2, 0, 1).contiguous()[None, :, :, :]  # FIXME still needed?
        #
        # transformed_image = input_image_torch
        # original_image_size = image.shape[:2]
        #
        # original_size = original_image_size
        # input_image = self.model.preprocess(transformed_image)
        # ######### MOVE END ############

        # ######### MOVE ############
        # # Transform input prompts
        # coords_torch, labels_torch, mask_input_torch = None, None, None
        #
        # points_torch = None
        # if point_coords is not None:
        #     assert (
        #         point_labels is not None
        #     ), "point_labels must be supplied if point_coords is supplied."
        #     point_coords = self.transform.apply_coords(point_coords, original_size)
        #     coords_torch = torch.as_tensor(point_coords, dtype=torch.float, device=self.device)
        #     labels_torch = torch.as_tensor(point_labels, dtype=torch.int, device=self.device)
        #     coords_torch, labels_torch = coords_torch[None, :, :], labels_torch[None, :]
        #     points_torch = (coords_torch, labels_torch)
        #
        # if mask_input is not None:  # TODO check with debugger if my masks are already good
        #     mask_input_torch = torch.as_tensor(mask_input, dtype=torch.float, device=self.device)
        #     mask_input_torch = mask_input_torch[None, :, :, :]
        #
        # ######### MOVE END ############

        # Embed images
        features = self.model.image_encoder(images)

        # Embed prompts
        point_labels = torch.ones_like(keypoints_xyc[:, :, -1])
        # visible_keypoints = keypoints_xyc[:, :, -1] > 0  # FIXME cannot remove non visible keypoints because will break batch dimension
        points = (keypoints_xyc[:, :, :-1], point_labels)
        sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
            points=points,
            masks=None,
            boxes=None
        )

        # Predict masks
        low_res_masks, iou_predictions = self.model.mask_decoder(  # FIXME DO NOT SUPPORT BATCH IMAGES
            image_embeddings=features,
            image_pe=self.model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )

        iou_predictions = iou_predictions[0].detach().cpu().numpy()
        low_res_masks = low_res_masks[0].detach().cpu().numpy()

        return low_res_masks


def _sam(sam):
    model = SamReID(sam)
    return model


def sam_vit_h(
    num_classes,
    loss="part_based",
    pretrained=True,
    enable_dim_reduction=True,
    dim_reduction_channels=256,
    pretrained_path="",
    **kwargs,
):
    sam = sam_model_registry["vit_h"](checkpoint=pretrained_path)
    return _sam(sam)


def sam_vit_l(
    num_classes,
    loss="part_based",
    pretrained=True,
    enable_dim_reduction=True,
    dim_reduction_channels=256,
    pretrained_path="",
    **kwargs,
):
    sam = sam_model_registry["vit_l"](checkpoint=pretrained_path)
    return _sam(sam)


def sam_vit_b(
    num_classes,
    loss="part_based",
    pretrained=True,
    enable_dim_reduction=True,
    dim_reduction_channels=256,
    pretrained_path="",
    **kwargs,
):
    sam = sam_model_registry["vit_b"](checkpoint=pretrained_path)
    return _sam(sam)
