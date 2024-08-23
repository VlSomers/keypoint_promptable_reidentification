import torch.nn as nn
import timm

from torchreid.models.promptable_transformer_backbone import PromptableTransformerBackbone


class ViT(PromptableTransformerBackbone):
    def __init__(self, name, num_classes, img_size, in_chans_masks, *args, **kwargs):
        model = timm.create_model(name, pretrained=True, num_classes=num_classes, global_pool='', img_size=img_size)
        print(model.default_cfg)
        patch_embed_size = model.patch_embed.grid_size
        masks_patch_embed = model.patch_embed.__class__(in_chans=in_chans_masks,
                                                        img_size=img_size,
                                                        patch_size=model.patch_embed.patch_size,
                                                        embed_dim=model.embed_dim,
                                                        bias=isinstance(model.norm_pre, nn.Identity))
        self.spatial_feature_shape = [patch_embed_size[0], patch_embed_size[1], model.embed_dim]
        super().__init__(model.patch_embed,
                         masks_patch_embed,
                         patch_embed_size,
                         patch_embed_dim=model.embed_dim,
                         feature_dim=model.embed_dim,
                         *args,
                         **kwargs)
        self.model = model

    def forward(self, images, prompt_masks=None, keypoints_xyc=None, cam_label=None, view_label=None, **kwargs):
        features = self.model.patch_embed(images)
        features = self._mask_embed(features, prompt_masks, images.shape[-2::])
        features = self.model._pos_embed(features)  # FIXME interpolate pos embedding?
        features = self._cam_embed(features, cam_label, view_label)
        features = self.model.patch_drop(features)
        features = self.model.norm_pre(features)
        features = self.model.blocks(features)
        features = self.model.norm(features)
        features = features[:, 1:, :]\
            .transpose(2, 1)\
            .unflatten(-1, self.model.patch_embed.grid_size)  # [N, D, Hf, Wf]
        return features


def timm_vit(
        name="",
        config=None,
        cam_num=0,
        view=0,
        num_classes=0,
        **kwargs,
):
    no_background_token =config.model.promptable_trans.no_background_token
    use_negative_keypoints = config.model.kpr.keypoints.use_negative_keypoints
    in_chans_masks = config.model.kpr.masks.prompt_parts_num
    if not no_background_token:
        in_chans_masks += 1
    if use_negative_keypoints:
        in_chans_masks += 1
    model = ViT(
        name=name,
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
        **kwargs,
    )
    return model


vit_timm_models = {
    "vit_base_patch16_224_miil": timm_vit,
    "vit_base_patch16_224": timm_vit,
    "vit_base_patch16_384": timm_vit,
    "vit_base_patch8_224": timm_vit,
    "samvit_base_patch16.sa1b": timm_vit,
}
