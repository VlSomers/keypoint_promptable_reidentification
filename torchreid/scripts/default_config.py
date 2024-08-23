import random
import uuid
from datetime import datetime
from yacs.config import CfgNode as CN
from torchreid.utils.constants import *
from deepdiff import DeepDiff
import re
import pprint

# Mapping from old to new configs:
# cfg.data.ro -> cfg.data.bipo
# cfg.data.transforms.ro -> cfg.data.transforms.bipo
# cfg.model.transreid.semantic_weight -> cfg.model.solider.semantic_weight
# cfg.model.transreid.test_weight -> cfg.model.solider.test_weight
# cfg.model.transreid.mask_path_emb_init_zeros -> cfg.model.solider.mask_path_emb_init_zeros
# cfg.model.vit.masks_prompting -> cfg.model.promptable_trans.masks_prompting
# cfg.model.vit.disable_inference_prompting -> cfg.model.promptable_trans.disable_inference_prompting
# cfg.model.vit.no_background_token -> cfg.model.promptable_trans.no_background_token
# cfg.model.vit.pose_encoding_strategy -> cfg.model.promptable_trans.pose_encoding_strategy
# cfg.model.vit.pose_encoding_all_layers -> cfg.model.promptable_trans.pose_encoding_all_layers
# cfg.model.vit.use_abs_pos_embed -> cfg.model.promptable_trans.use_abs_pos_embed
# cfg.model.vit.drop_path -> cfg.model.promptable_trans.drop_path
# cfg.model.vit.drop_out -> cfg.model.promptable_trans.drop_out
# cfg.model.vit.drop_rate -> cfg.model.promptable_trans.drop_rate
# cfg.model.vit.att_drop_rate -> cfg.model.promptable_trans.att_drop_rate
# cfg.model.vit.transformer_type -> cfg.model.promptable_trans.transformer_type
# cfg.model.vit.patch_size -> cfg.model.promptable_trans.patch_size
# cfg.model.vit.stride_size -> cfg.model.promptable_trans.stride_size

def get_default_config():
    cfg = CN()

    # project
    cfg.project = CN()
    cfg.project.name = "KPR"  # will be used as WanDB project name
    cfg.project.experiment_name = ""
    cfg.project.diff_config = ""
    cfg.project.notes = ""
    cfg.project.tags = []
    cfg.project.config_file = ""
    cfg.project.debug_mode = False
    cfg.project.logger = (
        CN()
    )  # Choose experiment manager client to use or simply use disk dump / matplotlib
    cfg.project.logger.use_clearml = False
    cfg.project.logger.use_tensorboard = False
    cfg.project.logger.use_wandb = False
    cfg.project.logger.matplotlib_show = False
    cfg.project.logger.save_disk = True  # save images to disk
    cfg.project.job_id = random.randint(0, 1_000_000_000)
    cfg.project.experiment_id = str(uuid.uuid4())
    cfg.project.start_time = datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%MS")

    # model
    cfg.model = CN()
    cfg.model.name = "kpr"
    cfg.model.compute_complexity = False
    cfg.model.pretrained = True  # automatically load pretrained model weights if available (For example HRNet
    # pretrained weights on ImageNet)
    cfg.model.load_weights = ""  # path to model weights, for doing inference with a model that was saved on disk with 'save_model_flag'
    cfg.model.load_config = (
        True  # load config saved with model weights and overwrite current config
    )
    cfg.model.backbone_pretrained_path = "pretrained_models/"  # path to pretrained weights for HRNet backbone, download on our Google Drive or on https://github.com/HRNet/HRNet-Image-Classification
    # number of horizontal stripes desired. When BPBreID is used, this variable will be automatically filled depending
    # on "data.masks.preprocess"
    cfg.model.discard_test_params = False  # if True, do not load test time config params from saved model
    cfg.model.resume = ""  # path to checkpoint for resume training
    cfg.model.save_model_flag = False  # path to checkpoint for resume training

    # configs for our part-based model BPBreID
    cfg.model.kpr = CN()
    cfg.model.kpr.spatial_feature_shape = []  # do not change
    cfg.model.kpr.pooling = "gwap"  # ['gap', 'gmp', 'gwap', 'gwap2']
    cfg.model.kpr.normalization = (
        "identity"  # ['identity', 'batch_norm_2d'] - obsolete, always use identity
    )
    cfg.model.kpr.mask_filtering_training = False  # use visibility scores at training - do not have an influence on testing performance yet, to be improved
    cfg.model.kpr.mask_filtering_testing = True  # use visibility scores at testing - do have a big influence on testing performance when activated
    cfg.model.kpr.last_stride = (
        1  # last stride of the resnet backbone - 1 for better performance
    )
    cfg.model.kpr.dim_reduce = "after_pooling"  #  where to apply feature dimensionality reduction (before or after global pooling) ['none', 'before_pooling', 'after_pooling', 'before_and_after_pooling', 'after_pooling_with_dropout']
    cfg.model.kpr.dim_reduce_output = (
        512  # reduce feature dimension to this value when above config is not 'none'
    )
    cfg.model.kpr.backbone = (
        "swinv2_base_window12to16_192to256.ms_in22k_ft_in1k"  # ['resnet50', 'hrnet32', 'fastreid_resnet_ibn_nl', 'solider_swin_base_patch4_window7_224']
    )
    cfg.model.kpr.learnable_attention_enabled = True  # use learnable attention mechanism to pool part features, otherwise, use fixed attention weights from external (pifpaf) heatmaps/masks
    cfg.model.kpr.test_embeddings = [
        "bn_foreg",
        "parts",
    ]  # embeddings to use at inference among ['globl', 'foreg', 'backg', 'conct', 'parts']: append 'bn_' suffix to use batch normed embeddings
    cfg.model.kpr.test_use_target_segmentation = "none"  # ['soft', 'hard', 'none'] - use external part mask to further refine the attention weights at inference
    cfg.model.kpr.training_binary_visibility_score = True  # use binary visibility score (0 or 1) instead of continuous visibility score (0 to 1) at training
    cfg.model.kpr.testing_binary_visibility_score = True  # use binary visibility score (0 or 1) instead of continuous visibility score (0 to 1) at testing
    cfg.model.kpr.use_prompt_visibility_score = False  # use visibility score derived from the prompt instead of learned attention maps
    cfg.model.kpr.enable_fpn = False  # use feature pyramid network (FPN)
    cfg.model.kpr.fpn_out_dim = 1024  # TODO fpn and msf # internal channel size for FPN
    cfg.model.kpr.enable_msf = True  # fuse multi stage feature maps to build high resolution feature maps from all stages of backbone. If disabled, spatial features of last stage are used
    cfg.model.kpr.msf_spatial_scale = -1  # spatial scale of the FPN: output feature map size = input image size / msf_spatial_scale. If set to -1, output feature map size = feature map size of backbone first stage
    cfg.model.kpr.shared_parts_id_classifier = False  # if each part branch uses share weights for the identity classifier. Used only when the identity loss is used on part-based embeddings.

    # Configs for the keypoints prompts given as input to KPR
    cfg.model.kpr.keypoints = CN()
    cfg.model.kpr.keypoints.enabled = True  # enable keypoint prompts in dataloader (load keypoints from disk, transforms and send to training engine)
    cfg.model.kpr.keypoints.vis_thresh = 0.3  # don't use keypoints with visibility score equal or below this threshold
    cfg.model.kpr.keypoints.vis_continous = False  # heatmaps derived from prompt also takes into account the visibility score for the gaussian peak value (only for prompts)
    cfg.model.kpr.keypoints.prompt_masks = "keypoints_gaussian"  # heuristic to generate prompt mask (heatmaps) from keypoints, {"keypoints", "keypoints_gaussian", "joints", "joints_gaussian"}
    cfg.model.kpr.keypoints.prompt_preprocess = "cck6"  # parts grouping strategy for prompt masks
    cfg.model.kpr.keypoints.target_masks = "none"  # heuristic to generate target human parsing labels from keypoints. Can be used if no PifPaf human parsing labels available{"keypoints", "keypoints_gaussian", "joints", "joints_gaussian"}
    cfg.model.kpr.keypoints.target_preprocess = "none"  # parts grouping strategy for target masks
    cfg.model.kpr.keypoints.use_negative_keypoints = True  # use keypoints from other non targets persons as negative prompts
    cfg.model.kpr.keypoints.kp_dir = "pifpaf_keypoints_pifpaf_maskrcnn_filtering"  # subdirectory where keypoints are stored
    cfg.model.kpr.keypoints.filter_out_no_skeletons = False  # remove training samples with no keypoints (+-203 for occluded duke)

    # Configs for the masks (target human parsing labels) used to train the body part attention head
    cfg.model.kpr.masks = CN()
    cfg.model.kpr.masks.enabled = True  # use masks loaded from disk or generated from heuristics like horizontal stripes
    cfg.model.kpr.masks.type = "disk"  # when 'disk' is used, load part masks from storage in 'cfg.model.kpr.masks.dir' folder  # TODO remove
    # when 'stripes' is used, divide the image in 'cfg.model.kpr.masks.parts_num' horizontal stripes in a PCB style.
    # 'stripes' with parts_num=1 can be used to emulate the global method Bag of Tricks (BoT)
    cfg.model.kpr.masks.parts_num = 1  # number of part-based embedding to extract. When PCB is used, change this parameter to the number of stripes required
    cfg.model.kpr.masks.parts_names = ["1"]   # do not change
    cfg.model.kpr.masks.prompt_parts_num = 1  # number of part-based embedding to extract. When PCB is used, change this parameter to the number of stripes required
    cfg.model.kpr.masks.prompt_parts_names = ["1"]  # do not change
    cfg.model.kpr.masks.dir = "pifpaf_maskrcnn_filtering"  # masks will be loaded from 'dataset_path/masks/<cfg.model.kpr.masks.dir>' directory
    cfg.model.kpr.masks.preprocess = "five_v"  # how to group the 36 pifpaf parts into smaller human semantic groups ['eight', 'five', 'four', 'two', ...], more combination available inside 'torchreid/data/masks_transforms/__init__.masks_preprocess_pifpaf'
    cfg.model.kpr.masks.softmax_weight = 15
    cfg.model.kpr.masks.background_computation_strategy = (
        "threshold"  # threshold, diff_from_max
    )
    cfg.model.kpr.masks.mask_filtering_threshold = 0.5

    # configs for transformers-based promptable backbones, affecting KPR when a transformer is used as backbone
    cfg.model.promptable_trans = CN()
    cfg.model.promptable_trans.drop_path = 0.1
    cfg.model.promptable_trans.drop_out = 0.0  # FIXME unused
    cfg.model.promptable_trans.drop_rate = 0.0
    cfg.model.promptable_trans.att_drop_rate = 0.0
    cfg.model.promptable_trans.transformer_type = 'vit_base_patch16_224_TransReID'
    cfg.model.promptable_trans.patch_size = 16
    cfg.model.promptable_trans.stride_size = [16, 16]  # [16, 16], stride: [12, 12]
    # configs for prompt tokenizer:
    cfg.model.promptable_trans.masks_prompting = True  # disable prompting mechanism, i.e. neither pass keypoint, nor seg masks prompts to KPR
    cfg.model.promptable_trans.disable_inference_prompting = False
    cfg.model.promptable_trans.no_background_token = False  # When True, remove (from the prompt) the background mask that was generated with the AddBackgroundMask transform.
    #   The background mask is the last channel in the prompt that contains a heatmaps of the background, i.e. heatmaps on regions not containing keypoint prompts.
    #   This config can probably always be set to True, since the background mask do not add any meaningful information to the prompt, and is just derived from the input keypoints.
    #   However, the provided models were (accidentally) trained with this background token, so it is set to False by default.
    cfg.model.promptable_trans.pose_encoding_strategy = 'embed_heatmaps_patches'  # 'spatialize_part_tokens' 'embed_heatmaps_patches'
    cfg.model.promptable_trans.pose_encoding_all_layers = False
    cfg.model.promptable_trans.use_abs_pos_embed = False

    # configs from TransReID, affecting KPR when a transformer is used as backbone
    cfg.model.transreid = CN()
    cfg.model.transreid.cam_num = 0  # TODO compute automatically from dataset
    cfg.model.transreid.jpm = True
    cfg.model.transreid.shift_num = 5
    cfg.model.transreid.shuffle_group = 2
    cfg.model.transreid.devide_length = 4
    cfg.model.transreid.re_arrange = True
    cfg.model.transreid.sie_coe = 3
    cfg.model.transreid.sie_camera = True
    cfg.model.transreid.sie_view = False

    # configs from SOLIDER, affecting KPR when SOLIDER is used as backbone
    cfg.model.solider = CN()
    cfg.model.solider.semantic_weight = 0.2
    cfg.model.solider.test_weight = ""
    cfg.model.solider.mask_path_emb_init_zeros = True

    # data
    cfg.data = CN()
    cfg.data.type = "image"
    cfg.data.root = "~/datasets/reid"
    cfg.data.sources = ["market1501"]
    cfg.data.targets = ["market1501"]
    cfg.data.workers = 4  # number of data loading workers, set to 0 to enable breakpoint debugging in dataloader code
    cfg.data.split_id = 0  # split index
    cfg.data.height = 256  # image height
    cfg.data.width = 128  # image width
    cfg.data.combineall = False  # combine train, query and gallery for training
    cfg.data.query_list = []  # list of query images to use, leave empty to use all
    cfg.data.transforms = [
        "rc",
        "re",
        "bipo",
    ]  # data augmentation from ['rf', 'rc', 're', 'cj'] = ['random flip', 'random crop', 'random erasing', 'color jitter']
    cfg.data.bipo = (
        CN()
    )  # parameters for random occlusion data augmentation with Pascal VOC, to be improved, not maintained
    cfg.data.bipo.path = ""
    cfg.data.bipo.p = 0.2
    cfg.data.bipo.n = 1
    cfg.data.bipo.masks_dir = "bpbreid_masks"  # directory where masks are stored
    cfg.data.bipo.min_overlap = 0.5
    cfg.data.bipo.max_overlap = 0.8
    cfg.data.bipo.pid_sampling_from_batch = True  # Occluders for the BIPO data augmentation are sampled from the current training batch. If set to False, occluders are sampled among the entire training set
    cfg.data.cj = CN()  # parameters for color jitter data augmentation
    cfg.data.cj.brightness = 0.2
    cfg.data.cj.contrast = 0.15
    cfg.data.cj.saturation = 0.0
    cfg.data.cj.hue = 0.0
    cfg.data.cj.always_apply = False
    cfg.data.cj.p = 0.5
    cfg.data.drk = CN()  # Drop random keypoints from input prompt
    cfg.data.drk.p = 0.2  # probability to drop a random keypoint
    cfg.data.drk.ratio = 0.5  # ratio of keypoints to drop
    cfg.data.dak = CN()  # Drop all keypoints from input prompt
    cfg.data.dak.p = 0.3  # probability to drop all keypoints
    cfg.data.resize = CN()
    cfg.data.resize.interpolation = 1  # 1 = INTER_LINEAR, 0 = INTER_NEAREST
    cfg.data.resize.mask_interpolation = 'bilinear'  # 'bilinear' or 'nearest' or 'nearest-exact'
    cfg.data.norm_mean = [0.485, 0.456, 0.406]  # default is imagenet mean
    cfg.data.norm_std = [0.229, 0.224, 0.225]  # default is imagenet std
    cfg.data.save_dir = "logs"  # save figures, images, logs, etc. in this folder
    cfg.data.load_train_targets = False
    cfg.data.mot = (
        CN()
    )  # Config for building a ReID dataset from a MOT dataset, not maintained
    cfg.data.mot.fig_size = (
        128,
        64,
    )  # Figure size for visualization purpose of the reid heatmaps/masks labels
    cfg.data.mot.mask_size = (
        32,
        16,
    )  # Size of saved the numpy array for the reid heatmaps/masks labels
    # For MOT challenge based datasets
    cfg.data.mot.train = CN()
    cfg.data.mot.train.min_vis = 0.3
    cfg.data.mot.train.min_h = 50
    cfg.data.mot.train.min_w = 25
    cfg.data.mot.train.min_samples_per_id = 4
    cfg.data.mot.train.max_samples_per_id = 40
    cfg.data.mot.train.max_total_ids = -1  # -1 means no limit
    cfg.data.mot.test = CN()
    cfg.data.mot.test.min_vis = 0.3
    cfg.data.mot.test.min_h = 50
    cfg.data.mot.test.min_w = 25
    cfg.data.mot.test.min_samples_per_id = 4
    cfg.data.mot.test.max_samples_per_id = 40
    cfg.data.mot.test.max_total_ids = -1  # -1 means no limit
    cfg.data.mot.test.ratio_query_per_id = 0.2

    # specific datasets
    cfg.market1501 = CN()
    cfg.market1501.use_500k_distractors = (
        False  # add 500k distractors to the gallery set for market1501
    )
    cfg.cuhk03 = CN()
    cfg.cuhk03.labeled_images = (
        False  # use labeled images, if False, use detected images
    )
    cfg.cuhk03.classic_split = False  # use classic split by Li et al. CVPR14
    cfg.cuhk03.use_metric_cuhk03 = False  # use cuhk03's metric for evaluation

    cfg.occluded_posetrack = CN()
    cfg.occluded_posetrack.enable_dataset_sampling_loading = True
    cfg.occluded_posetrack.occluded_dataset = True
    cfg.occluded_posetrack.enable_sam = True  # pifpaf body part masks are too coarse (overlap background) and cover all humans in the bbox. Compute a SAM segmentation mask with the pifpaf keypoints of the target person as prompt, and only keep pif and paf field inside that SAM ask.
    cfg.occluded_posetrack.sam_checkpoint = "~/pretrained_models/sam/sam_vit_h_4b8939.pth"

    # sampler
    cfg.sampler = CN()
    cfg.sampler.train_sampler = (
        "RandomIdentitySampler"  # sampler for source train loader
    )
    cfg.sampler.train_sampler_t = (
        "RandomIdentitySampler"  # sampler for target train loader
    )
    cfg.sampler.num_instances = (
        4  # number of instances per identity for RandomIdentitySampler
    )

    # video reid setting
    cfg.video = CN()
    cfg.video.seq_len = 15  # number of images to sample in a tracklet
    cfg.video.sample_method = (
        "evenly"  # how to sample images from a tracklet 'random'/'evenly'/'all'
    )
    cfg.video.pooling_method = "avg"  # how to pool features over a tracklet

    # train
    cfg.train = CN()
    cfg.train.optim = "sgd"
    cfg.train.lr = 0.008
    cfg.train.reduced_lr = 0.0002
    cfg.train.weight_decay = 1e-4
    cfg.train.weight_decay_bias = 1e-4  # used for TransReID param_groups construction
    cfg.train.warmup_t = 5  # for cosine lr scheduler
    cfg.train.max_epoch = 120
    cfg.train.start_epoch = 0
    cfg.train.batch_size = 64
    cfg.train.fixbase_epoch = 0  # number of epochs to fix base layers. When the Solider backbone is employed, this will only affect the prompt tokenizer, and freeze it for this number of epochs.
    cfg.train.open_layers = []  # layers for training while keeping others frozen, e.g. ["classifier"]
    cfg.train.staged_lr = False  # set different lr to different layers
    cfg.train.new_layers = []  # newly added layers with default lr, e.g. ["classifier"]
    cfg.train.base_lr_mult = 2.0  # learning rate multiplier for base layers
    cfg.train.lr_scheduler = "cosine_annealing_warmup"
    cfg.train.stepsize = [40, 70]  # stepsize to decay learning rate
    cfg.train.gamma = 0.1  # learning rate decay multiplier
    cfg.train.seed = 1  # random seed
    cfg.train.eval_freq = (
        -1
    )  # evaluation frequency (-1 means to only test after training)
    cfg.train.batch_debug_freq = 0
    cfg.train.batch_log_freq = 0
    cfg.train.mixed_precision = True  # Use Torch automatic mixed precision package as in TransReID
    cfg.train.transreid_lr = True  # to apply TransReID param_groups construction

    # optimizer
    cfg.sgd = CN()
    cfg.sgd.momentum = 0.9  # momentum factor for sgd and rmsprop
    cfg.sgd.dampening = 0.0  # dampening for momentum
    cfg.sgd.nesterov = False  # Nesterov momentum
    cfg.rmsprop = CN()
    cfg.rmsprop.alpha = 0.99  # smoothing constant
    cfg.adam = CN()
    cfg.adam.beta1 = 0.9  # exponential decay rate for first moment
    cfg.adam.beta2 = 0.999  # exponential decay rate for second moment

    # loss
    cfg.loss = CN()
    cfg.loss.name = (
        "part_based"  # use part based engine to train kpr with GiLt loss
    )
    cfg.loss.part_based = CN()
    cfg.loss.part_based.name = "part_averaged_triplet_loss"  # ['inter_parts_triplet_loss', 'intra_parts_triplet_loss', 'part_max_triplet_loss', 'part_averaged_triplet_loss', 'part_min_triplet_loss', 'part_max_min_triplet_loss', 'part_random_max_min_triplet_loss']
    cfg.loss.part_based.ppl = "cl"  # body part prediction loss: ['cl', 'fl', 'dl'] = [cross entropy loss with label smoothing, focal loss, dice loss]
    cfg.loss.part_based.best_pred_ratio = 1.0  # Only <best_pred_ratio>% best predicted pixels will contribute to the total body part prediction loss. This is to avoid the model being penalized too much for not predicting the coarse human parsing labels.
    cfg.loss.part_based.weights = (
        CN()
    )  # weights to apply for the different losses and different types of embeddings, for more details, have a look at 'torchreid/losses/GiLt_loss.py'
    cfg.loss.part_based.weights[GLOBAL] = CN()
    cfg.loss.part_based.weights[GLOBAL].id = 1.0
    cfg.loss.part_based.weights[GLOBAL].tr = 0.0
    cfg.loss.part_based.weights[FOREGROUND] = CN()
    cfg.loss.part_based.weights[FOREGROUND].id = 1.0
    cfg.loss.part_based.weights[FOREGROUND].tr = 0.0
    cfg.loss.part_based.weights[CONCAT_PARTS] = CN()
    cfg.loss.part_based.weights[CONCAT_PARTS].id = 1.0
    cfg.loss.part_based.weights[CONCAT_PARTS].tr = 0.0
    cfg.loss.part_based.weights[PARTS] = CN()
    cfg.loss.part_based.weights[PARTS].id = 0.0
    cfg.loss.part_based.weights[PARTS].tr = 1.0
    cfg.loss.part_based.weights[PIXELS] = CN()
    cfg.loss.part_based.weights[PIXELS].ce = 0.35
    cfg.loss.softmax = CN()
    cfg.loss.softmax.label_smooth = True  # use label smoothing regularizer
    cfg.loss.triplet = CN()
    cfg.loss.triplet.margin = 0.3  # distance margin
    cfg.loss.triplet.weight_t = 1.0  # weight to balance hard triplet loss
    cfg.loss.triplet.weight_x = 0.0  # weight to balance cross entropy loss

    # test
    cfg.test = CN()
    cfg.test.batch_size = 128
    cfg.test.batch_size_pairwise_dist_matrix = 500  # query to gallery distance matrix is computed on the GPU by batch of gallery samples with this size.
    # To avoid out of memory issue, we don't compute it for all gallery samples at the same time, but we compute it
    # in batches of 'batch_size_pairwise_dist_matrix' gallery samples.
    cfg.test.dist_metric = "euclidean"  # distance metric, ['euclidean', 'cosine']
    cfg.test.normalize_feature = (
        True  # normalize feature vectors before computing distance
    )
    cfg.test.ranks = [1, 3, 5, 10, 20]  # cmc ranks
    cfg.test.evaluate = False  # test only
    cfg.test.start_eval = 0  # start to evaluate after a specific epoch
    cfg.test.rerank = False  # use person re-ranking
    cfg.test.visrank = (
        True  # visualize ranked results (only available when cfg.test.evaluate=True)
    )
    cfg.test.visrank_topk = 10  # top-k ranks to visualize
    cfg.test.visrank_count = 10  # number of top-k ranks to plot
    cfg.test.visrank_display_mode = 'display_worst_rand'  # 'display_worst' or 'display_worst_rand' to randomly display middle and worst performing queries
    cfg.test.visrank_q_idx_list = []  # list of ids of queries for which we want to plot topk rank. If len(visrank_q_idx_list) < visrank_count, remaining ids will be random
    cfg.test.vis_feature_maps = False
    cfg.test.visrank_per_body_part = False
    cfg.test.vis_embedding_projection = False
    cfg.test.save_features = False  # save test set extracted features to disk
    cfg.test.detailed_ranking = (
        False  # display ranking performance for each part individually
    )
    cfg.test.part_based = CN()
    cfg.test.part_based.dist_combine_strat = "mean"  # ['mean', 'max'] local part based distances are combined into a global distance using this strategy

    # inference
    cfg.inference = CN()
    cfg.inference.enabled = False
    cfg.inference.input_folder = ""
    cfg.inference.output_folder = ""
    cfg.inference.output_figure_folder = ""

    return cfg


keys_to_ignore_in_diff = {
    "cfg.project",
    "cfg.model.save_model_flag",
    "cfg.model.kpr.backbone",
    "cfg.model.kpr.learnable_attention_enabled",
    "cfg.model.kpr.masks.parts_num",
    "cfg.model.kpr.masks.dir",
    "cfg.data.type",
    "cfg.data.root",
    "cfg.data.sources",
    "cfg.data.targets",
    "cfg.data.workers",
    "cfg.data.split_id",
    "cfg.data.combineall",
    "cfg.data.save_dir",
    "cfg.train.eval_freq",
    "cfg.train.batch_debug_freq",
    "cfg.train.batch_log_freq",
    "cfg.test.batch_size",
    "cfg.test.batch_size_pairwise_dist_matrix",
    "cfg.test.dist_metric",
    "cfg.test.ranks",
    "cfg.test.evaluate",
    "cfg.test.start_eval",
    "cfg.test.rerank",
    "cfg.test.visrank",
    "cfg.test.visrank_topk",
    "cfg.test.visrank_count",
    "cfg.test.visrank_q_idx_list",
    "cfg.test.vis_feature_maps",
    "cfg.test.visrank_per_body_part",
    "cfg.test.vis_embedding_projection",
    "cfg.test.save_features",
    "cfg.test.detailed_ranking",
    "cfg.train.open_layers",
    "cfg.model.load_weights",
}


def imagedata_kwargs(cfg):
    return {
        "config": cfg,
        "root": cfg.data.root,
        "sources": cfg.data.sources,
        "targets": cfg.data.targets,
        "height": cfg.data.height,
        "width": cfg.data.width,
        "transforms": cfg.data.transforms,
        "norm_mean": cfg.data.norm_mean,
        "norm_std": cfg.data.norm_std,
        "use_gpu": cfg.use_gpu,
        "split_id": cfg.data.split_id,
        "combineall": cfg.data.combineall,
        "load_train_targets": cfg.data.load_train_targets,
        "batch_size_train": cfg.train.batch_size,
        "batch_size_test": cfg.test.batch_size,
        "workers": cfg.data.workers,
        "num_instances": cfg.sampler.num_instances,
        "train_sampler": cfg.sampler.train_sampler,
        "train_sampler_t": cfg.sampler.train_sampler_t,
        # image
        "cuhk03_labeled": cfg.cuhk03.labeled_images,
        "cuhk03_classic_split": cfg.cuhk03.classic_split,
        "market1501_500k": cfg.market1501.use_500k_distractors,
        "masks_dir": cfg.model.kpr.masks.dir,
    }


def videodata_kwargs(cfg):
    return {
        "root": cfg.data.root,
        "sources": cfg.data.sources,
        "targets": cfg.data.targets,
        "height": cfg.data.height,
        "width": cfg.data.width,
        "transforms": cfg.data.transforms,
        "norm_mean": cfg.data.norm_mean,
        "norm_std": cfg.data.norm_std,
        "use_gpu": cfg.use_gpu,
        "split_id": cfg.data.split_id,
        "combineall": cfg.data.combineall,
        "batch_size_train": cfg.train.batch_size,
        "batch_size_test": cfg.test.batch_size,
        "workers": cfg.data.workers,
        "num_instances": cfg.sampler.num_instances,
        "train_sampler": cfg.sampler.train_sampler,
        # video
        "seq_len": cfg.video.seq_len,
        "sample_method": cfg.video.sample_method,
    }


def optimizer_kwargs(cfg):
    return {
        "optim": cfg.train.optim,
        "lr": cfg.train.lr,
        "reduced_lr": cfg.train.reduced_lr,
        "weight_decay": cfg.train.weight_decay,
        "weight_decay_bias": cfg.train.weight_decay_bias,
        "momentum": cfg.sgd.momentum,
        "sgd_dampening": cfg.sgd.dampening,
        "sgd_nesterov": cfg.sgd.nesterov,
        "rmsprop_alpha": cfg.rmsprop.alpha,
        "adam_beta1": cfg.adam.beta1,
        "adam_beta2": cfg.adam.beta2,
        "staged_lr": cfg.train.staged_lr,
        "new_layers": cfg.train.new_layers,
        "base_lr_mult": cfg.train.base_lr_mult,
        "transreid_lr": cfg.train.transreid_lr,
    }


def lr_scheduler_kwargs(cfg):
    return {
        "lr_scheduler": cfg.train.lr_scheduler,
        "stepsize": cfg.train.stepsize,
        "gamma": cfg.train.gamma,
        "max_epoch": cfg.train.max_epoch,
        "lr": cfg.train.lr,
        "warmup_t": cfg.train.warmup_t,
    }


def engine_run_kwargs(cfg):
    return {
        "save_dir": cfg.data.save_dir,
        "fixbase_epoch": cfg.train.fixbase_epoch,
        "open_layers": cfg.train.open_layers,
        "test_only": cfg.test.evaluate,
        "dist_metric": cfg.test.dist_metric,
        "normalize_feature": cfg.test.normalize_feature,
        "visrank": cfg.test.visrank,
        "visrank_topk": cfg.test.visrank_topk,
        "visrank_q_idx_list": cfg.test.visrank_q_idx_list,
        "visrank_count": cfg.test.visrank_count,
        "use_metric_cuhk03": cfg.cuhk03.use_metric_cuhk03,
        "ranks": cfg.test.ranks,
        "rerank": cfg.test.rerank,
        "save_features": cfg.test.save_features,
    }


def display_config_diff(cfg, default_cfg_copy):
    def iterdict(d):
        for k, v in d.items():
            if isinstance(v, dict):
                iterdict(v)
            else:
                if type(v) == list:
                    v = str(v)
                d.update({k: v})
        return d

    ddiff = DeepDiff(
        iterdict(default_cfg_copy), iterdict(cfg.clone()), ignore_order=True
    )
    cfg_diff = {}
    if "values_changed" in ddiff:
        for k, v in ddiff["values_changed"].items():
            reformatted_key = "cfg." + k.replace("root['", "").replace(
                "']['", "."
            ).replace("']", "")
            if "[" in reformatted_key:
                reformatted_key = reformatted_key.split("[")[0]
            reformatted_key_split = reformatted_key.split(".")
            ignore_key = False
            for i in range(2, len(reformatted_key_split) + 1):
                prefix = ".".join(reformatted_key_split[0:i])
                if prefix in keys_to_ignore_in_diff:
                    ignore_key = True
                    break
            if not ignore_key:
                key = re.findall(r"\['([A-Za-z0-9_]+)'\]", k)[-1]
                cfg_diff[key] = v["new_value"]
    print("Diff from default config :")
    pprint.pprint(cfg_diff)
    if len(str(cfg_diff)) < 128:
        cfg.project.diff_config = str(cfg_diff)
    else:
        cfg.project.diff_config = str(cfg_diff)[0:124] + "..."
