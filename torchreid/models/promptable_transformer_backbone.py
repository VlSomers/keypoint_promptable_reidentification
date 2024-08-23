from collections import OrderedDict

import torch
import math
from torch import nn as nn
from torch.nn import functional as F

from torchreid.models.kpr import AfterPoolingDimReduceLayer

class PromptableTransformerBackbone(nn.Module):
    """ class to be inherited by all promptable transformer backbones.
    It defines how prompt should be tokenized (i.e. the implementation of the prompt tokenizer).
    It also defines how camera information should be embedded similar to Transreid.
    """
    def __init__(self,
                 patch_embed,
                 masks_patch_embed,
                 patch_embed_size,
                 config,
                 patch_embed_dim,
                 feature_dim,
                 use_negative_keypoints=False,
                 camera=0,
                 view=0,
                 sie_xishu =1.0,
                 masks_prompting=False,
                 disable_inference_prompting=False,
                 prompt_parts_num=0,
                 **kwargs,
                 ):
        super().__init__()

        # standard params
        self.feature_dim = self.num_features = feature_dim  # num_features for consistency with other models
        self.patch_embed_dim = patch_embed_dim

        # prompt related params
        self.masks_prompting = masks_prompting
        self.disable_inference_prompting = disable_inference_prompting
        self.prompt_parts_num = prompt_parts_num
        self.pose_encoding_strategy = config.pose_encoding_strategy
        self.pose_encoding_all_layers = config.pose_encoding_all_layers
        self.no_background_token = config.no_background_token
        self.use_negative_keypoints = use_negative_keypoints

        # patch embedding for image and prompt
        self.patch_embed_size = patch_embed_size
        self.patch_embed = patch_embed
        self.masks_patch_embed = masks_patch_embed
        self.num_patches = self.patch_embed_size[0] * self.patch_embed_size[1]

        # token for camera and view
        self.cam_num = camera
        self.view_num = view
        self.sie_xishu = sie_xishu

        # Initialize SIE Embedding
        if camera > 1 and view > 1:
            self.sie_embed = nn.Parameter(torch.zeros(camera * view, 1, self.patch_embed_dim))
            trunc_normal_(self.sie_embed, std=.02)
            print('camera number is : {} and viewpoint number is : {}'.format(camera, view))
            print('using SIE_Lambda is : {}'.format(sie_xishu))
        elif camera > 1:
            self.sie_embed = nn.Parameter(torch.zeros(camera, 1, self.patch_embed_dim))
            trunc_normal_(self.sie_embed, std=.02)
            print('camera number is : {}'.format(camera))
            print('using SIE_Lambda is : {}'.format(sie_xishu))
        elif view > 1:
            self.sie_embed = nn.Parameter(torch.zeros(view, 1, self.patch_embed_dim))
            trunc_normal_(self.sie_embed, std=.02)
            print('viewpoint number is : {}'.format(view))
            print('using SIE_Lambda is : {}'.format(sie_xishu))
        else:
            self.sie_embed = None

        # token for parts
        self.num_part_tokens = self.prompt_parts_num + 1
        if self.use_negative_keypoints:
            self.num_part_tokens += 1
        self.parts_embed = nn.Parameter(torch.zeros(self.num_part_tokens, 1, self.patch_embed_dim))  # +1 for background
        self.num_layers = 4  # FIXME
        if self.pose_encoding_all_layers:
            self.parts_embed_dim_upscales = nn.ModuleDict({str(self.patch_embed_dim * 2 ** i) : AfterPoolingDimReduceLayer(self.patch_embed_dim, self.patch_embed_dim * 2 ** i) for i in range(self.num_layers-1)})

        # init tokens
        trunc_normal_(self.parts_embed, std=.02)

    def _cam_embed(self, images, cam_label, view_label):
        reshape = False
        if len(images.shape) == 4:
            b, h, w, c = images.shape
            images = images.view(b, h * w, c)
            reshape = True
        if self.cam_num > 0 and self.view_num > 0:
            images = images + self.sie_xishu * self.sie_embed[cam_label * self.view_num + view_label]
        elif self.cam_num > 0:
            images = images + self.sie_xishu * self.sie_embed[cam_label]
        elif self.view_num > 0:
            images = images + self.sie_xishu * self.sie_embed[view_label]
        else:
            images = images
        if reshape:
            images = images.view(b, h, w, c)
        return images

    """The Prompt Tokenizer, to tokenize the input keypoint prompt information and add it to images tokens.
    Here, keypoints prompts in the (x, y, c) format are already pre-processed (see 'torchreid/data/datasets/dataset.py -> ImageDataset.getitem()') 
    and turned into dense heatmaps of shape (K+2, H, W) where K is the number of parts, and K+2 include the negative keypoints and the background, and H, W are the height and width of the image.
    'prompt_masks' is therefore a tensor of shape (B, K+2, H, W) where B is the batch size."""
    def _mask_embed(self, image_features, prompt_masks, input_size):
        if self.masks_prompting:
            if prompt_masks is not None and prompt_masks.shape[2:] != input_size:
                prompt_masks = F.interpolate(
                    prompt_masks,
                    size=input_size,
                    mode="bilinear",
                    align_corners=True
                )
            if self.disable_inference_prompting or prompt_masks is None:
                prompt_masks = torch.zeros([image_features.shape[0], self.num_part_tokens, input_size[0], input_size[1]], device=image_features.device)
                if not self.no_background_token:
                    prompt_masks[:, 0] = 1.  # if the background token was enabled when training the model, the "empty"
                    # prompts the model has seen during training are prompts with 0 filled heatmaps on each channel, and 1 filled heatmap on the background channel.
                    # The model should therefore be prompted with a similar empty prompt during inference when prompts are disabled.
            prompt_masks = prompt_masks.type(image_features.dtype)
            if self.pose_encoding_strategy == 'embed_heatmaps_patches':
                prompt_masks.requires_grad = False  # should be unecessary
                if self.no_background_token:
                    prompt_masks = prompt_masks[:, 1:]  # remove background mask that was generated with the AddBackgroundMask transform
                part_tokens = self.masks_patch_embed(prompt_masks)
                part_tokens = part_tokens[0] if isinstance(part_tokens, tuple) else part_tokens
            elif self.pose_encoding_strategy == 'spatialize_part_tokens':  # TODO add another variant where token multiplied by continuous mask
                parts_embed = self.parts_embed
                if parts_embed.shape[-1] != image_features.shape[-1]:
                    parts_embed = self.parts_embed_dim_upscales[str(image_features.shape[-1])](parts_embed)
                prompt_masks.requires_grad = False
                parts_segmentation_map = prompt_masks.argmax(
                    dim=1)  # map each patch to a part index (or background)
                part_tokens = parts_embed[parts_segmentation_map].squeeze(-2)  # map each patch to a part token
                if self.no_background_token:
                    part_tokens[parts_segmentation_map == 0] = 0  # FIXME if no_background_token, make the background token a non learnable/zero vector
                    # TODO if no background token, add only part_token where necessary, with images[parts_segmentation_map] += part_tokens[parts_segmentation_map]
                if len(part_tokens.shape) != len(image_features.shape):
                    part_tokens = part_tokens.flatten(start_dim=1, end_dim=2)
            else:
                raise NotImplementedError
            image_features += part_tokens
        return image_features

    def _combine_layers(self, features, layers, prompt_masks):  # TODO remove unused
        features_per_stage = OrderedDict()
        for i, layer in enumerate(layers):
            features_size = features.shape[-2::]
            features = layer(features)
            if self.pose_encoding_all_layers:  # TODO make it work in other scenarios/configs
                self._mask_embed(features, prompt_masks, features_size)
            features_per_stage[i] = features.permute(0, 3, 1, 2)
        return features_per_stage


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # type: (Tensor, float, float, float, float) -> Tensor
    r"""Fills the input Tensor with values drawn from a truncated
    normal distribution. The values are effectively drawn from the
    normal distribution :math:`\mathcal{N}(\text{mean}, \text{std}^2)`
    with values outside :math:`[a, b]` redrawn until they are within
    the bounds. The method used for generating the random values works
    best when :math:`a \leq \text{mean} \leq b`.
    Args:
        tensor: an n-dimensional `torch.Tensor`
        mean: the mean of the normal distribution
        std: the standard deviation of the normal distribution
        a: the minimum cutoff value
        b: the maximum cutoff value
    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.trunc_normal_(w)
    """
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        print("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",)

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor
