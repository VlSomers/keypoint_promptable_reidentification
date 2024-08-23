from torchreid.models.transreid.configs.defaults import get_default_transreid_config
from .make_model import make_model, __factory_T_type


def merge_with_torchreid_config(transreid_cfg, cfg):
    transreid_cfg.MODEL.PRETRAIN_PATH = cfg.model.backbone_pretrained_path
    # Transformer setting
    transreid_cfg.MODEL.DROP_PATH = cfg.model.promptable_trans.drop_path
    transreid_cfg.MODEL.DROP_OUT = cfg.model.promptable_trans.drop_out
    transreid_cfg.MODEL.ATT_DROP_RATE = cfg.model.promptable_trans.att_drop_rate
    transreid_cfg.MODEL.TRANSFORMER_TYPE = cfg.model.promptable_trans.transformer_type
    transreid_cfg.MODEL.STRIDE_SIZE = cfg.model.promptable_trans.stride_size
    # JPM Parameter
    transreid_cfg.MODEL.JPM = cfg.model.transreid.jpm
    transreid_cfg.MODEL.SHIFT_NUM = cfg.model.transreid.shift_num
    transreid_cfg.MODEL.SHUFFLE_GROUP = cfg.model.transreid.shuffle_group
    transreid_cfg.MODEL.DEVIDE_LENGTH = cfg.model.transreid.devide_length
    transreid_cfg.MODEL.RE_ARRANGE = cfg.model.transreid.re_arrange
    # SIE Parameter
    transreid_cfg.MODEL.SIE_COE = cfg.model.transreid.sie_coe
    transreid_cfg.MODEL.SIE_CAMERA = cfg.model.transreid.sie_camera
    transreid_cfg.MODEL.SIE_VIEW = cfg.model.transreid.sie_view
    # Input size
    transreid_cfg.INPUT.SIZE_TRAIN = [cfg.data.height, cfg.data.width]
    transreid_cfg.INPUT.SIZE_TEST = [cfg.data.height, cfg.data.width]
    return transreid_cfg


def transreid(
    num_classes,
    loss="part_based",
    pretrained=True,
    enable_dim_reduction=True,
    dim_reduction_channels=256,
    pretrained_path="",
    config=None,
    use_as_backbone=False,
    **kwargs,
):
    view_num = 1
    camera_num = config.model.transreid.cam_num
    transreid_cfg = get_default_transreid_config()
    transreid_cfg = merge_with_torchreid_config(transreid_cfg, config)
    model = make_model(transreid_cfg, use_as_backbone=use_as_backbone, num_class=num_classes, camera_num=camera_num, view_num=view_num)
    return model
