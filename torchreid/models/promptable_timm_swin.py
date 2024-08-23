from collections import OrderedDict

import numpy as np
import torch.nn as nn
import timm

from timm.layers import PatchEmbed

from torchreid.models.promptable_transformer_backbone import PromptableTransformerBackbone


class SwinTransformer(PromptableTransformerBackbone):
    def __init__(self, name, num_classes, img_size, in_chans_masks, enable_fpn, *args, **kwargs):
        model = timm.create_model(name,
                                  pretrained=True,
                                  num_classes=num_classes,
                                  global_pool='',
                                  img_size=img_size,
                                  )
        print(model.default_cfg)
        patch_embed_size = model.patch_embed.grid_size
        masks_patch_embed = PatchEmbed(
            in_chans=in_chans_masks,
            img_size=img_size,
            patch_size=model.patch_embed.patch_size,
            embed_dim=model.embed_dim,
            norm_layer=model.patch_embed.norm.__class__ if not isinstance(model.patch_embed.norm, nn.Identity) else None,
            output_fmt='NHWC',
        )
        self.enable_fpn = enable_fpn
        self.spatial_feature_depth_per_layer = np.array([inf["num_chs"] for inf in model.feature_info])
        if self.enable_fpn:
            self.spatial_feature_depth = self.spatial_feature_depth_per_layer.sum()
            self.spatial_feature_shape = [int(img_size[0] / model.feature_info[0]['reduction']),
                                          int(img_size[1] / model.feature_info[0]['reduction']),
                                          self.spatial_feature_depth]
        else:
            self.spatial_feature_depth = model.feature_info[-1]['num_chs']
            self.spatial_feature_shape = [int(img_size[0] / model.feature_info[-1]['reduction']),
                                          int(img_size[1] / model.feature_info[-1]['reduction']),
                                          self.spatial_feature_depth]
        super().__init__(model.patch_embed,
                         masks_patch_embed,
                         patch_embed_size,
                         patch_embed_dim=model.embed_dim,
                         feature_dim=model.num_features,
                         *args,
                         **kwargs
                         )
        self.model = model
        self.norm = nn.LayerNorm(self.spatial_feature_shape[-1])
        # feature pyramid network to build high resolution semantic feature maps from multiple stages:

    def forward(self, images, prompt_masks=None, keypoints_xyc=None, cam_label=None, view_label=None, **kwargs):
        features = self.model.patch_embed(images)
        if cam_label is not None or view_label is not None:
            features = self._cam_embed(features, cam_label, view_label)
        features_per_stage = OrderedDict()
        for i, layer in enumerate(self.model.layers):
            if i == 0 or self.pose_encoding_all_layers:  # TODO make it work in other scenarios/configs
                features = self._mask_embed(features, prompt_masks, images.shape[-2::])
            features = layer(features)
            features_per_stage[i] = features.permute(0, 3, 1, 2)

        if self.enable_fpn:
            features = features_per_stage
        else:
            features = features_per_stage[list(features_per_stage.keys())[-1]]  # last layer output
            features = self.norm(features.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)  # TODO apply it also after FPN?
        return features



def timm_swin(
        name="",
        config=None,
        cam_num=0,
        view=0,
        num_classes=0,
        enable_fpn=True,
        **kwargs,
):
    no_background_token = config.model.promptable_trans.no_background_token
    use_negative_keypoints = config.model.kpr.keypoints.use_negative_keypoints
    in_chans_masks = config.model.kpr.masks.prompt_parts_num
    if not no_background_token:
        in_chans_masks += 1
    if use_negative_keypoints:
        in_chans_masks += 1
    model = SwinTransformer(
        name=name,
        pretrained_model="",
        config=config.model.promptable_trans,
        num_classes=0,
        use_negative_keypoints=config.model.kpr.keypoints.use_negative_keypoints,
        img_size=[config.data.height, config.data.width],
        in_chans_masks=in_chans_masks,
        camera=cam_num if config.model.transreid.sie_camera else 0,
        view=view if config.model.transreid.sie_view else 0,
        sie_xishu=config.model.transreid.sie_coe,
        masks_prompting=config.model.promptable_trans.masks_prompting,
        disable_inference_prompting=config.model.promptable_trans.disable_inference_prompting,
        prompt_parts_num=config.model.kpr.masks.prompt_parts_num,
        enable_fpn=enable_fpn,
        **kwargs,
    )
    return model


swin_timm_models = {
    "swin_base_patch4_window12_384.ms_in1k": timm_swin,
    "swin_large_patch4_window12_384.ms_in22k_ft_in1k": timm_swin,
    'swinv2_base_window12to24_192to384.ms_in22k_ft_in1k': timm_swin,
    'swinv2_base_window12to16_192to256.ms_in22k_ft_in1k': timm_swin,
    'swinv2_base_window8_256.ms_in1k': timm_swin,
    'swinv2_base_window16_256.ms_in1k': timm_swin,
    'swinv2_base_window12_192.ms_in22k': timm_swin,
    'swin_base_patch4_window7_224.ms_in22k_ft_in1k': timm_swin,
}
