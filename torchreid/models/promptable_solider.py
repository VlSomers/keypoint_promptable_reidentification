import os
import torch
import numpy as np

from torchreid.models.promptable_transformer_backbone import PromptableTransformerBackbone
from torchreid.models.solider.backbones.swin_transformer import SwinTransformer, PatchEmbed


class PromptableSoliderSwinTransformer(PromptableTransformerBackbone):
    def __init__(self,
                 img_size,
                 in_chans_masks,
                 enable_fpn,
                 pretrained_path,
                 pretrained_model,
                 pretrained,
                 drop_path,
                 drop_rate,
                 att_drop_rate,
                 semantic_weight,
                 convert_weights,
                 mask_path_emb_init_zeros,
                 *args,
                 **kwargs):

        embed_dim=128
        patch_size=4
        strides = (4, 2, 2, 2)
        norm_cfg = dict(type='LN')
        patch_norm = True
        spatial_reduce = 32
        pretrained_path = os.path.join(pretrained_path, pretrained_model)
        model = SwinTransformer(
            pretrain_img_size=img_size,
            patch_size=patch_size,
            window_size=7,
            embed_dims=embed_dim,
            depths=(2, 2, 18, 2),
            strides=strides,
            norm_cfg=norm_cfg,
            num_heads=(4, 8, 16, 32),
            drop_path_rate=drop_path,
            drop_rate=drop_rate,
            attn_drop_rate=att_drop_rate,
            convert_weights=convert_weights,
            patch_norm=patch_norm,
            semantic_weight=semantic_weight,
            pretrained=pretrained_path,
            # **kwargs
        )
        if pretrained:
            model.init_weights(pretrained_path)
        patch_embed_size = np.array(img_size) / patch_size  # FIXME do not work with strides

        masks_patch_embed = PatchEmbed(
            in_channels=in_chans_masks,
            embed_dims=embed_dim,
            conv_type='Conv2d',
            kernel_size=patch_size,
            stride=strides[0],
            norm_cfg=norm_cfg if patch_norm else None,
            init_cfg=None
        )

        if mask_path_emb_init_zeros:
            masks_patch_embed.zero_weights()

        # masks_patch_embed = PatchEmbedTimm(
        #     img_size=img_size,
        #     patch_size=patch_size,
        #     in_chans=in_chans_masks,
        #     embed_dim=embed_dim,
        #     norm_layer=nn.LayerNorm,
        #     # output_fmt='NHWC',
        # )

        self.enable_fpn = enable_fpn
        self.spatial_feature_depth_per_layer = np.array(model.num_features)
        if self.enable_fpn:
            self.spatial_feature_depth = self.spatial_feature_depth_per_layer.sum()
            self.spatial_feature_shape = [int(img_size[0] / patch_size),
                                          int(img_size[1] / patch_size),
                                          self.spatial_feature_depth]
        else:
            self.spatial_feature_depth = self.spatial_feature_depth_per_layer[-1]
            self.spatial_feature_shape = [int(img_size[0] / spatial_reduce),
                                          int(img_size[1] / spatial_reduce),
                                          self.spatial_feature_depth]  # [64 32 1920]
        super().__init__(model.patch_embed,
                         masks_patch_embed,
                         patch_embed_size,
                         patch_embed_dim=embed_dim,
                         feature_dim=model.num_features,
                         *args,
                         **kwargs
                         )
        self.model = model
        # self.norm = nn.LayerNorm(self.spatial_feature_shape[-1])
        # feature pyramid network to build high resolution semantic feature maps from multiple stages:

    # def forward(self, images, prompt_masks=None, keypoints_xyc=None, cam_label=None, view_label=None, **kwargs):
    #     features = self.model.patch_embed(images)
    #     features = self._mask_embed(features, prompt_masks)
    #     if cam_label is not None or view_label is not None:
    #         features = self._cam_embed(features, cam_label, view_label)
    #     if self.enable_fpn:
    #         features = self._combine_layers(features, self.model.layers)
    #     else:
    #         features = self.model.layers(features)
    #         features = self.model.norm(features)
    #         features = features.permute(0, 3, 1, 2)
    #     return features

    def forward(self, images, semantic_weight=None, prompt_masks=None, keypoints_xyc=None, cam_label=None, view_label=None, **kwargs):
        if self.model.semantic_weight >= 0 and semantic_weight == None:
            w = torch.ones(images.shape[0],1) * self.model.semantic_weight
            w = torch.cat([w, 1-w], axis=-1)
            semantic_weight = w.to(images.device)

        features, hw_shape = self.model.patch_embed(images)
        features = self._mask_embed(features, prompt_masks, images.shape[-2::])

        if cam_label is not None or view_label is not None:
            features = self._cam_embed(features, cam_label, view_label)

        if self.model.use_abs_pos_embed:
            features = features + self.model.absolute_pos_embed
        features = self.model.drop_after_pos(features)

        outs = []
        for i, stage in enumerate(self.model.stages):
            features, hw_shape, out, out_hw_shape = stage(features, hw_shape)
            if self.model.semantic_weight >= 0:
                sw = self.model.semantic_embed_w[i](semantic_weight).unsqueeze(1)
                sb = self.model.semantic_embed_b[i](semantic_weight).unsqueeze(1)
                features = features * self.model.softplus(sw) + sb
            if i in self.model.out_indices:
                norm_layer = getattr(self.model, f'norm{i}')
                out = norm_layer(out)
                out = out.view(-1, *out_hw_shape, self.model.num_features[i]).permute(0, 3, 1, 2).contiguous()
                outs.append(out)
        if self.enable_fpn:
            return {i: feat for i, feat in enumerate(outs)}
        else:
            return outs[-1]


def solider_swin(
        config=None,
        cam_num=0,
        view=0,
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

    model = PromptableSoliderSwinTransformer(
        pretrained_model='SOLIDER/swin_base_reid.pth',
        convert_weights=False,
        use_abs_pos_embed=config.model.promptable_trans.use_abs_pos_embed,
        config=config.model.promptable_trans,
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
        drop_path=config.model.promptable_trans.drop_path,
        drop_rate=config.model.promptable_trans.drop_rate,
        att_drop_rate=config.model.promptable_trans.att_drop_rate,
        semantic_weight=config.model.solider.semantic_weight,
        mask_path_emb_init_zeros=config.model.solider.mask_path_emb_init_zeros,
        **kwargs,
    )
    return model


def imagenet_swin(
        config=None,
        cam_num=0,
        view=0,
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

    model = PromptableSoliderSwinTransformer(
        pretrained_model='SOLIDER/swin_base_patch4_window7_224_22k.pth',
        convert_weights=True,
        use_abs_pos_embed=config.model.promptable_trans.use_abs_pos_embed,
        config=config.model.promptable_trans,
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
        drop_path=config.model.promptable_trans.drop_path,
        drop_rate=config.model.promptable_trans.drop_rate,
        att_drop_rate=config.model.promptable_trans.att_drop_rate,
        semantic_weight=config.model.solider.semantic_weight,
        mask_path_emb_init_zeros=config.model.solider.mask_path_emb_init_zeros,
        **kwargs,
    )
    return model


solider_models = {
    "solider_swin_base_patch4_window7_224": solider_swin,
    "imagenet_swin_base_patch4_window7_224": imagenet_swin,
}