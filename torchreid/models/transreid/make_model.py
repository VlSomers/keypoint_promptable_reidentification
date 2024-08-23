import os

import torch
import torch.nn as nn
import copy
from .backbones.vit_pytorch import vit_base_patch16_224_TransReID, vit_small_patch16_224_TransReID, deit_small_patch16_224_TransReID
from ...utils.constants import GLOBAL, BACKGROUND, FOREGROUND, CONCAT_PARTS, PARTS, BN_GLOBAL, BN_BACKGROUND, \
    BN_FOREGROUND, BN_CONCAT_PARTS, BN_PARTS


def shuffle_unit(features, shift, group, begin=1):

    batchsize = features.size(0)
    dim = features.size(-1)
    # Shift Operation
    feature_random = torch.cat([features[:, begin-1+shift:], features[:, begin:begin-1+shift]], dim=1)
    x = feature_random
    # Patch Shuffle Operation
    try:
        x = x.view(batchsize, group, -1, dim)
    except:
        x = torch.cat([x, x[:, -2:-1, :]], dim=1)
        x = x.view(batchsize, group, -1, dim)

    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(batchsize, -1, dim)

    return x


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)

    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)


class build_transformer(nn.Module):
    def __init__(self, use_as_backbone, num_classes, camera_num, view_num, cfg, factory, model_filename):
        super(build_transformer, self).__init__()
        last_stride = cfg.MODEL.LAST_STRIDE
        model_path = os.path.join(cfg.MODEL.PRETRAIN_PATH, model_filename[cfg.MODEL.TRANSFORMER_TYPE])
        model_name = cfg.MODEL.NAME
        pretrain_choice = cfg.MODEL.PRETRAIN_CHOICE
        self.cos_layer = cfg.MODEL.COS_LAYER
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT
        self.in_planes = 768
        self.use_as_backbone = use_as_backbone

        print('using Transformer_type: {} as a backbone'.format(cfg.MODEL.TRANSFORMER_TYPE))

        if cfg.MODEL.SIE_CAMERA:
            camera_num = camera_num
        else:
            camera_num = 0
        if cfg.MODEL.SIE_VIEW:
            view_num = view_num
        else:
            view_num = 0

        self.base = factory[cfg.MODEL.TRANSFORMER_TYPE](img_size=cfg.INPUT.SIZE_TRAIN, sie_xishu=cfg.MODEL.SIE_COE,
                                                        camera=camera_num, view=view_num, stride_size=cfg.MODEL.STRIDE_SIZE, drop_path_rate=cfg.MODEL.DROP_PATH,
                                                        drop_rate= cfg.MODEL.DROP_OUT,
                                                        attn_drop_rate=cfg.MODEL.ATT_DROP_RATE)
        if cfg.MODEL.TRANSFORMER_TYPE == 'deit_small_patch16_224_TransReID':
            self.in_planes = 384
        if pretrain_choice == 'imagenet':
            self.base.load_param(model_path)
            print('Loading pretrained ImageNet model......from {}'.format(model_path))

        self.gap = nn.AdaptiveAvgPool2d(1)

        self.num_classes = num_classes
        self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)
        self.spatial_feature_shape = [self.base.patch_embed.num_y, self.base.patch_embed.num_x, self.base.num_features]

    def forward(self, x, label=None, cam_label=None, view_label=None, **kwargs):
        global_feat = self.base(x, cam_label=cam_label, view_label=view_label)

        feat = self.bottleneck(global_feat)

        # if self.training:
        #     # if self.ID_LOSS_TYPE in ('arcface', 'cosface', 'amsoftmax', 'circle'):
        #     #     cls_score = self.classifier(feat, label)
        #     # else:
        #     #     cls_score = self.classifier(feat)
        #     cls_score = self.classifier(feat)
        #     return cls_score, global_feat  # global feature for triplet loss
        # else:
        #     if self.neck_feat == 'after':
        #         # print("Test with feature after BN")
        #         return feat
        #     else:
        #         # print("Test with feature before BN")
        #         return global_feat


        if self.training:
            cls_score = self.classifier(feat)
            # Outputs
            embeddings = {
                GLOBAL: global_feat,  # [N, D]
                BACKGROUND: None,  # [N, D]
                FOREGROUND: None,  # [N, D]
                CONCAT_PARTS: None,  # [N, K*D]
                PARTS: None,  # [N, K, D]
                BN_GLOBAL: feat,  # [N, D]
                BN_BACKGROUND: None,  # [N, D]
                BN_FOREGROUND: None,  # [N, D]
                BN_CONCAT_PARTS: None,  # [N, K*D]
                BN_PARTS: None,  # [N, K, D]
            }

            visibility_scores = {
                GLOBAL: torch.ones(global_feat.shape[0], device=global_feat.device, dtype=torch.bool),  # [N]
                BACKGROUND: None,  # [N]
                FOREGROUND: None,  # [N]
                CONCAT_PARTS: None,  # [N]
                PARTS: None,  # [N, K]
            }

            id_cls_scores = {
                GLOBAL: cls_score,  # [N, num_classes]
                BACKGROUND: None,  # [N, num_classes]
                FOREGROUND: None,  # [N, num_classes]
                CONCAT_PARTS: None,  # [N, num_classes]
                PARTS: None,  # [N, K, num_classes]
            }

            masks = {
                GLOBAL: None,
                BACKGROUND: None,
                FOREGROUND: None,
                CONCAT_PARTS: None,
                PARTS: None,
            }

            pixels_cls_scores = None
            spatial_features = None

            return embeddings, visibility_scores, id_cls_scores, pixels_cls_scores, spatial_features, masks
        else:
            # Outputs
            embeddings = {
                GLOBAL: global_feat,  # [N, D]
                BACKGROUND: None,  # [N, D]
                FOREGROUND: None,  # [N, D]
                CONCAT_PARTS: None,  # [N, K*D]
                PARTS: None,  # [N, K, D]
                BN_GLOBAL: feat,  # [N, D]
                BN_BACKGROUND: None,  # [N, D]
                BN_FOREGROUND: None,  # [N, D]
                BN_CONCAT_PARTS: None,  # [N, K*D]
                BN_PARTS: None,
                # [N, K, D]
            }

            visibility_scores = {
                GLOBAL: torch.ones(global_feat.shape[0], device=global_feat.device, dtype=torch.bool),  # [N]
                BACKGROUND: None,  # [N]
                FOREGROUND: None,  # [N]
                CONCAT_PARTS: None,  # [N]
                PARTS: None,  # [N, K]
            }

            id_cls_scores = {
                GLOBAL: None,  # [N, num_classes]
                BACKGROUND: None,  # [N, num_classes]
                FOREGROUND: None,  # [N, num_classes]
                CONCAT_PARTS: None,  # [N, num_classes]
                PARTS: None,  # [N, K, num_classes]
            }

            masks = {
                GLOBAL: torch.ones((global_feat.shape[0], 32, 16), device=global_feat.device),  # [N, Hf, Wf]
                BACKGROUND: None,  # [N, Hf, Wf]
                FOREGROUND: None,  # [N, Hf, Wf]
                CONCAT_PARTS: None,  # [N, Hf, Wf]
                PARTS: None,  # [N, K, Hf, Wf]
            }

            pixels_cls_scores = None
            spatial_features = None

            return embeddings, visibility_scores, id_cls_scores, pixels_cls_scores, spatial_features, masks

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))

    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model for finetuning from {}'.format(model_path))


class build_transformer_local(nn.Module):
    def __init__(self, use_as_backbone, num_classes, camera_num, view_num, cfg, factory, model_filename, rearrange):
        super(build_transformer_local, self).__init__()
        self.feature_dim = 768
        model_path = os.path.join(cfg.MODEL.PRETRAIN_PATH, model_filename[cfg.MODEL.TRANSFORMER_TYPE])
        pretrain_choice = cfg.MODEL.PRETRAIN_CHOICE
        self.cos_layer = cfg.MODEL.COS_LAYER
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT
        self.in_planes = 768
        self.use_as_backbone = use_as_backbone

        print('using Transformer_type: {} as a backbone'.format(cfg.MODEL.TRANSFORMER_TYPE))

        if cfg.MODEL.SIE_CAMERA:
            camera_num = camera_num
        else:
            camera_num = 0

        if cfg.MODEL.SIE_VIEW:
            view_num = view_num
        else:
            view_num = 0

        self.base = factory[cfg.MODEL.TRANSFORMER_TYPE](img_size=cfg.INPUT.SIZE_TRAIN, sie_xishu=cfg.MODEL.SIE_COE, local_feature=cfg.MODEL.JPM, camera=camera_num, view=view_num, stride_size=cfg.MODEL.STRIDE_SIZE, drop_path_rate=cfg.MODEL.DROP_PATH)

        if pretrain_choice == 'imagenet':
            self.base.load_param(model_path)
            print('Loading pretrained ImageNet model......from {}'.format(model_path))

        block = self.base.blocks[-1]
        layer_norm = self.base.norm
        self.b1 = nn.Sequential(  # global branch with one transformer layer
            copy.deepcopy(block),
            copy.deepcopy(layer_norm)
        )
        self.b2 = nn.Sequential(  # jpm branch with one transformer layer
            copy.deepcopy(block),
            copy.deepcopy(layer_norm)
        )

        self.num_classes = num_classes

        self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)
        self.classifier_1 = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier_1.apply(weights_init_classifier)
        self.classifier_2 = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier_2.apply(weights_init_classifier)
        self.classifier_3 = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier_3.apply(weights_init_classifier)
        self.classifier_4 = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier_4.apply(weights_init_classifier)

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)
        self.bottleneck_1 = nn.BatchNorm1d(self.in_planes)
        self.bottleneck_1.bias.requires_grad_(False)
        self.bottleneck_1.apply(weights_init_kaiming)
        self.bottleneck_2 = nn.BatchNorm1d(self.in_planes)
        self.bottleneck_2.bias.requires_grad_(False)
        self.bottleneck_2.apply(weights_init_kaiming)
        self.bottleneck_3 = nn.BatchNorm1d(self.in_planes)
        self.bottleneck_3.bias.requires_grad_(False)
        self.bottleneck_3.apply(weights_init_kaiming)
        self.bottleneck_4 = nn.BatchNorm1d(self.in_planes)
        self.bottleneck_4.bias.requires_grad_(False)
        self.bottleneck_4.apply(weights_init_kaiming)

        self.shuffle_groups = cfg.MODEL.SHUFFLE_GROUP
        print('using shuffle_groups size:{}'.format(self.shuffle_groups))
        self.shift_num = cfg.MODEL.SHIFT_NUM
        print('using shift_num size:{}'.format(self.shift_num))
        self.divide_length = cfg.MODEL.DEVIDE_LENGTH
        print('using divide_length size:{}'.format(self.divide_length))
        self.rearrange = rearrange

        self.spatial_feature_shape = [self.base.patch_embed.num_y, self.base.patch_embed.num_x, self.base.num_features]


    def forward(self, x, label=None, cam_label= None, view_label=None, **kwargs):  # label is unused if self.cos_layer == 'no'

        features = self.base(x, cam_label=cam_label, view_label=view_label)

        if self.use_as_backbone:
            # return spatial feature map without cls token
            return features[:, 1:, :].transpose(2, 1).unflatten(-1, (self.base.patch_embed.num_y, self.base.patch_embed.num_x))  # [N, D, Hf, Wf]

        # global branch
        b1_feat = self.b1(features) # [64, 129, 768]
        global_feat = b1_feat[:, 0]

        # JPM branch
        feature_length = features.size(1) - 1
        patch_length = feature_length // self.divide_length
        token = features[:, 0:1]  # clss token that we reuse for each jigsaw patch and then perform loss on it

        if self.rearrange:  # True
            x = shuffle_unit(features, self.shift_num, self.shuffle_groups)
        else:
            x = features[:, 1:]
        # lf_1
        b1_local_feat = x[:, :patch_length]
        b1_local_feat = self.b2(torch.cat((token, b1_local_feat), dim=1))
        local_feat_1 = b1_local_feat[:, 0]

        # lf_2
        b2_local_feat = x[:, patch_length:patch_length*2]
        b2_local_feat = self.b2(torch.cat((token, b2_local_feat), dim=1))
        local_feat_2 = b2_local_feat[:, 0]

        # lf_3
        b3_local_feat = x[:, patch_length*2:patch_length*3]
        b3_local_feat = self.b2(torch.cat((token, b3_local_feat), dim=1))
        local_feat_3 = b3_local_feat[:, 0]

        # lf_4
        b4_local_feat = x[:, patch_length*3:patch_length*4]
        b4_local_feat = self.b2(torch.cat((token, b4_local_feat), dim=1))
        local_feat_4 = b4_local_feat[:, 0]

        feat = self.bottleneck(global_feat)

        local_feat_1_bn = self.bottleneck_1(local_feat_1)
        local_feat_2_bn = self.bottleneck_2(local_feat_2)
        local_feat_3_bn = self.bottleneck_3(local_feat_3)
        local_feat_4_bn = self.bottleneck_4(local_feat_4)

        cls_score = self.classifier(feat)
        cls_score_1 = self.classifier_1(local_feat_1_bn)
        cls_score_2 = self.classifier_2(local_feat_2_bn)
        cls_score_3 = self.classifier_3(local_feat_3_bn)
        cls_score_4 = self.classifier_4(local_feat_4_bn)


        if self.training:
            # Outputs
            embeddings = {
                GLOBAL: global_feat,  # [N, D]
                BACKGROUND: None,  # [N, D]
                FOREGROUND: None,  # [N, D]
                CONCAT_PARTS: None,  # [N, K*D]
                PARTS: torch.stack([local_feat_1, local_feat_2, local_feat_3, local_feat_4], dim=1),  # [N, K, D]
                BN_GLOBAL: feat,  # [N, D]
                BN_BACKGROUND: None,  # [N, D]
                BN_FOREGROUND: None,  # [N, D]
                BN_CONCAT_PARTS: None,  # [N, K*D]
                BN_PARTS: torch.stack([local_feat_1_bn, local_feat_2_bn, local_feat_3_bn, local_feat_4_bn], dim=1),  # [N, K, D]
            }

            visibility_scores = {
                GLOBAL: torch.ones(local_feat_1_bn.shape[0], device=local_feat_1_bn.device, dtype=torch.bool),  # [N]
                BACKGROUND: None,  # [N]
                FOREGROUND: None,  # [N]
                CONCAT_PARTS: None,  # [N]
                PARTS: torch.ones((embeddings[PARTS].shape[0], embeddings[PARTS].shape[1]), device=local_feat_1_bn.device, dtype=torch.bool),  # [N, K]
            }

            id_cls_scores = {
                GLOBAL: cls_score,  # [N, num_classes]
                BACKGROUND: None,  # [N, num_classes]
                FOREGROUND: None,  # [N, num_classes]
                CONCAT_PARTS: None,  # [N, num_classes]
                PARTS: torch.stack([cls_score_1, cls_score_2, cls_score_3, cls_score_4], dim=1),  # [N, K, num_classes]
            }

            masks = {
                GLOBAL: None,
                BACKGROUND: None,
                FOREGROUND: None,
                CONCAT_PARTS: None,
                PARTS: None,
            }

            pixels_cls_scores = None
            spatial_features = None

            return embeddings, visibility_scores, id_cls_scores, pixels_cls_scores, spatial_features, masks
        else:
            # Outputs
            embeddings = {
                GLOBAL: torch.stack([global_feat, local_feat_1 / 4, local_feat_2 / 4, local_feat_3 / 4, local_feat_4 / 4], dim=1),  # [N, D]
                BACKGROUND: None,  # [N, D]
                FOREGROUND: None,  # [N, D]
                CONCAT_PARTS: None,  # [N, K*D]
                PARTS: None,  # [N, K, D]
                BN_GLOBAL: feat,  # [N, D]
                BN_BACKGROUND: None,  # [N, D]
                BN_FOREGROUND: None,  # [N, D]
                BN_CONCAT_PARTS: None,  # [N, K*D]
                BN_PARTS: None,
                # [N, K, D]
            }

            visibility_scores = {
                GLOBAL: torch.ones(local_feat_1_bn.shape[0], device=local_feat_1_bn.device, dtype=torch.bool),  # [N]
                BACKGROUND: None,  # [N]
                FOREGROUND: None,  # [N]
                CONCAT_PARTS: None,  # [N]
                PARTS: None,  # [N, K]
            }

            id_cls_scores = {
                GLOBAL: None,  # [N, num_classes]
                BACKGROUND: None,  # [N, num_classes]
                FOREGROUND: None,  # [N, num_classes]
                CONCAT_PARTS: None,  # [N, num_classes]
                PARTS: None,  # [N, K, num_classes]
            }

            masks = {
                GLOBAL: torch.ones((local_feat_1_bn.shape[0], 32, 16), device=local_feat_1_bn.device),  # [N, Hf, Wf]
                BACKGROUND: None,  # [N, Hf, Wf]
                FOREGROUND: None,  # [N, Hf, Wf]
                CONCAT_PARTS: None,  # [N, Hf, Wf]
                PARTS: None,  # [N, K, Hf, Wf]
            }

            pixels_cls_scores = None
            spatial_features = None

            return embeddings, visibility_scores, id_cls_scores, pixels_cls_scores, spatial_features, masks

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))

    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model for finetuning from {}'.format(model_path))


__factory_T_type = {
    'vit_base_patch16_224_TransReID': vit_base_patch16_224_TransReID,
    'deit_base_patch16_224_TransReID': vit_base_patch16_224_TransReID,
    'vit_small_patch16_224_TransReID': vit_small_patch16_224_TransReID,
    'deit_small_patch16_224_TransReID': deit_small_patch16_224_TransReID
}

__model_filename = {
    'vit_base_patch16_224_TransReID': "jx_vit_base_p16_224-80ecf9dd.pth",
    'deit_base_patch16_224_TransReID': "",
    'vit_small_patch16_224_TransReID': "",
    'deit_small_patch16_224_TransReID': ""
}


def make_model(cfg, use_as_backbone, num_class, camera_num, view_num):
    if cfg.MODEL.JPM:
        model = build_transformer_local(use_as_backbone, num_class, camera_num, view_num, cfg, __factory_T_type, __model_filename, rearrange=cfg.MODEL.RE_ARRANGE)
        print('===========building transformer with JPM module ===========')
    else:
        model = build_transformer(use_as_backbone, num_class, camera_num, view_num, cfg, __factory_T_type, __model_filename)
        print('===========building transformer===========')
    return model
