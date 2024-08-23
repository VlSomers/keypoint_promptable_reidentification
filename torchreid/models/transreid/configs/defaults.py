from yacs.config import CfgNode as CN

# -----------------------------------------------------------------------------
# Convention about Training / Test specific parameters
# -----------------------------------------------------------------------------
# Whenever an argument can be either used for training or for testing, the
# corresponding name will be post-fixed by a _TRAIN for a training parameter,

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------

def get_default_transreid_config():
    cfg = CN()
    # -----------------------------------------------------------------------------
    # MODEL
    # -----------------------------------------------------------------------------
    cfg.MODEL = CN()
    # Using cuda or cpu for training
    cfg.MODEL.DEVICE = "cuda"
    # ID number of GPU
    cfg.MODEL.DEVICE_ID = '0'
    # Name of backbone
    cfg.MODEL.NAME = 'transformer'
    # Last stride of backbone
    cfg.MODEL.LAST_STRIDE = 1
    # Path to pretrained model of backbone
    cfg.MODEL.PRETRAIN_PATH = ''
    
    # Use ImageNet pretrained model to initialize backbone or use self trained model to initialize the whole model
    # Options: 'imagenet' , 'self' , 'finetune'
    cfg.MODEL.PRETRAIN_CHOICE = 'imagenet'
    
    # If train with BNNeck, options: 'bnneck' or 'no'
    cfg.MODEL.NECK = 'bnneck'
    # If train loss include center loss, options: 'yes' or 'no'. Loss with center loss has different optimizer configuration
    cfg.MODEL.IF_WITH_CENTER = 'no'
    
    cfg.MODEL.ID_LOSS_TYPE = 'softmax'
    cfg.MODEL.ID_LOSS_WEIGHT = 1.0
    cfg.MODEL.TRIPLET_LOSS_WEIGHT = 1.0
    
    cfg.MODEL.METRIC_LOSS_TYPE = 'triplet'
    # If train with multi-gpu ddp mode, options: 'True', 'False'
    cfg.MODEL.DIST_TRAIN = False
    # If train with soft triplet loss, options: 'True', 'False'
    cfg.MODEL.NO_MARGIN = False
    # If train with label smooth, options: 'on', 'off'
    cfg.MODEL.IF_LABELSMOOTH = 'on'
    # If train with arcface loss, options: 'True', 'False'
    cfg.MODEL.COS_LAYER = False
    
    # Transformer setting
    cfg.MODEL.DROP_PATH = 0.1
    cfg.MODEL.DROP_OUT = 0.0
    cfg.MODEL.ATT_DROP_RATE = 0.0
    cfg.MODEL.TRANSFORMER_TYPE = 'None'
    cfg.MODEL.STRIDE_SIZE = [16, 16]
    
    # JPM Parameter
    cfg.MODEL.JPM = False
    cfg.MODEL.SHIFT_NUM = 5
    cfg.MODEL.SHUFFLE_GROUP = 2
    cfg.MODEL.DEVIDE_LENGTH = 4
    cfg.MODEL.RE_ARRANGE = True
    
    # SIE Parameter
    cfg.MODEL.SIE_COE = 3.0
    cfg.MODEL.SIE_CAMERA = False
    cfg.MODEL.SIE_VIEW = False
    
    # -----------------------------------------------------------------------------
    # INPUT
    # -----------------------------------------------------------------------------
    cfg.INPUT = CN()
    # Size of the image during training
    cfg.INPUT.SIZE_TRAIN = [384, 128]
    # Size of the image during test
    cfg.INPUT.SIZE_TEST = [384, 128]
    # Random probability for image horizontal flip
    cfg.INPUT.PROB = 0.5
    # Random probability for random erasing
    cfg.INPUT.RE_PROB = 0.5
    # Values to be used for image normalization
    cfg.INPUT.PIXEL_MEAN = [0.485, 0.456, 0.406]
    # Values to be used for image normalization
    cfg.INPUT.PIXEL_STD = [0.229, 0.224, 0.225]
    # Value of padding size
    cfg.INPUT.PADDING = 10
    
    # -----------------------------------------------------------------------------
    # Dataset
    # -----------------------------------------------------------------------------
    cfg.DATASETS = CN()
    # List of the dataset names for training, as present in paths_catalog.py
    cfg.DATASETS.NAMES = ('market1501')
    # Root directory where datasets should be used (and downloaded if not found)
    cfg.DATASETS.ROOT_DIR = ('../data')
    
    
    # -----------------------------------------------------------------------------
    # DataLoader
    # -----------------------------------------------------------------------------
    cfg.DATALOADER = CN()
    # Number of data loading threads
    cfg.DATALOADER.NUM_WORKERS = 8
    # Sampler for data loading
    cfg.DATALOADER.SAMPLER = 'softmax'
    # Number of instance for one batch
    cfg.DATALOADER.NUM_INSTANCE = 16
    
    # ---------------------------------------------------------------------------- #
    # Solver
    # ---------------------------------------------------------------------------- #
    cfg.SOLVER = CN()
    # Name of optimizer
    cfg.SOLVER.OPTIMIZER_NAME = "Adam"
    # Number of max epoches
    cfg.SOLVER.MAX_EPOCHS = 100
    # Base learning rate
    cfg.SOLVER.BASE_LR = 3e-4
    # Whether using larger learning rate for fc layer
    cfg.SOLVER.LARGE_FC_LR = False
    # Factor of learning bias
    cfg.SOLVER.BIAS_LR_FACTOR = 2
    # Factor of learning bias
    cfg.SOLVER.SEED = 1234
    # Momentum
    cfg.SOLVER.MOMENTUM = 0.9
    # Margin of triplet loss
    cfg.SOLVER.MARGIN = 0.3
    # Learning rate of SGD to learn the centers of center loss
    cfg.SOLVER.CENTER_LR = 0.5
    # Balanced weight of center loss
    cfg.SOLVER.CENTER_LOSS_WEIGHT = 0.0005
    
    # Settings of weight decay
    cfg.SOLVER.WEIGHT_DECAY = 0.0005
    cfg.SOLVER.WEIGHT_DECAY_BIAS = 0.0005
    
    # decay rate of learning rate
    cfg.SOLVER.GAMMA = 0.1
    # decay step of learning rate
    cfg.SOLVER.STEPS = (40, 70)
    # warm up factor
    cfg.SOLVER.WARMUP_FACTOR = 0.01
    #  warm up epochs
    cfg.SOLVER.WARMUP_EPOCHS = 5
    # method of warm up, option: 'constant','linear'
    cfg.SOLVER.WARMUP_METHOD = "linear"
    
    cfg.SOLVER.COSINE_MARGIN = 0.5
    cfg.SOLVER.COSINE_SCALE = 30
    
    # epoch number of saving checkpoints
    cfg.SOLVER.CHECKPOINT_PERIOD = 10
    # iteration of display training log
    cfg.SOLVER.LOG_PERIOD = 100
    # epoch number of validation
    cfg.SOLVER.EVAL_PERIOD = 10
    # Number of images per batch
    # This is global, so if we have 8 GPUs and IMS_PER_BATCH = 128, each GPU will
    # contain 16 images per batch
    cfg.SOLVER.IMS_PER_BATCH = 64
    
    # ---------------------------------------------------------------------------- #
    # TEST
    # ---------------------------------------------------------------------------- #
    
    cfg.TEST = CN()
    # Number of images per batch during test
    cfg.TEST.IMS_PER_BATCH = 128
    # If test with re-ranking, options: 'True','False'
    cfg.TEST.RE_RANKING = False
    # Path to trained model
    cfg.TEST.WEIGHT = ""
    # Which feature of BNNeck to be used for test, before or after BNNneck, options: 'before' or 'after'
    cfg.TEST.NECK_FEAT = 'after'
    # Whether feature is nomalized before test, if yes, it is equivalent to cosine distance
    cfg.TEST.FEAT_NORM = 'yes'
    
    # Name for saving the distmat after testing.
    cfg.TEST.DIST_MAT = "dist_mat.npy"
    # Whether calculate the eval score option: 'True', 'False'
    cfg.TEST.EVAL = False
    # ---------------------------------------------------------------------------- #
    # Misc options
    # ---------------------------------------------------------------------------- #
    # Path to checkpoint and saved log of trained model
    cfg.OUTPUT_DIR = ""

    return cfg