# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.
from detectron2.config import CfgNode as CN


def add_mask_former_default_config(cfg):
    # data config
    # select the dataset mapper
    cfg.INPUT.DATASET_MAPPER_NAME = "mask_former_semantic"
    # Color augmentation
    cfg.INPUT.COLOR_AUG_SSD = False
    # We retry random cropping until no single category in semantic segmentation GT occupies more
    # than `SINGLE_CATEGORY_MAX_AREA` part of the crop.
    cfg.INPUT.CROP.SINGLE_CATEGORY_MAX_AREA = 1.0
    # Pad image and segmentation GT in dataset mapper.
    cfg.INPUT.SIZE_DIVISIBILITY = -1

    # solver config
    # weight decay on embedding
    cfg.SOLVER.WEIGHT_DECAY_EMBED = 0.0
    # optimizer
    cfg.SOLVER.OPTIMIZER = "ADAMW"
    cfg.SOLVER.BACKBONE_MULTIPLIER = 0.1

    cfg.SOLVER.MODE = 'prompt'

    # mask_former model config
    cfg.MODEL.MASK_FORMER = CN()

    # loss
    cfg.MODEL.MASK_FORMER.DEEP_SUPERVISION = True
    cfg.MODEL.MASK_FORMER.NO_OBJECT_WEIGHT = 0.1
    cfg.MODEL.MASK_FORMER.CLASS_WEIGHT = 1.0
    cfg.MODEL.MASK_FORMER.DICE_WEIGHT = 1.0
    cfg.MODEL.MASK_FORMER.MASK_WEIGHT = 20.0

    # transformer config
    cfg.MODEL.MASK_FORMER.NHEADS = 8
    cfg.MODEL.MASK_FORMER.DROPOUT = 0.1
    cfg.MODEL.MASK_FORMER.DIM_FEEDFORWARD = 2048
    cfg.MODEL.MASK_FORMER.ENC_LAYERS = 0
    cfg.MODEL.MASK_FORMER.DEC_LAYERS = 6
    cfg.MODEL.MASK_FORMER.PRE_NORM = False

    cfg.MODEL.MASK_FORMER.HIDDEN_DIM = 256
    cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES = 156

    cfg.MODEL.MASK_FORMER.TRANSFORMER_IN_FEATURE = "res5"
    cfg.MODEL.MASK_FORMER.ENFORCE_INPUT_PROJ = False

    # mask_former inference config
    cfg.MODEL.MASK_FORMER.TEST = CN()
    cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON = False
    cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON = False
    cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON = False
    cfg.MODEL.MASK_FORMER.TEST.OBJECT_MASK_THRESHOLD = 0.0
    cfg.MODEL.MASK_FORMER.TEST.OVERLAP_THRESHOLD = 0.0
    cfg.MODEL.MASK_FORMER.TEST.SEM_SEG_POSTPROCESSING_BEFORE_INFERENCE = False

    # you can use this config to override
    cfg.MODEL.MASK_FORMER.SIZE_DIVISIBILITY = 32

    cfg.MODEL.EXTRA = CN()
    cfg.MODEL.EXTRA.AUX_LOSS = True
    cfg.MODEL.EXTRA.KPT_LOSS_COEF = 5.0
    cfg.MODEL.EXTRA.DEC_LAYERS = 10
    cfg.MODEL.EXTRA.EOS_COEF = 0.1


    # pixel decoder config
    cfg.MODEL.SEM_SEG_HEAD.MASK_DIM = 256
    # adding transformer in pixel decoder
    cfg.MODEL.SEM_SEG_HEAD.TRANSFORMER_ENC_LAYERS = 0
    # pixel decoder
    cfg.MODEL.SEM_SEG_HEAD.PIXEL_DECODER_NAME = "BasePixelDecoder"

    # swin transformer backbone
    cfg.MODEL.SWIN = CN()
    cfg.MODEL.SWIN.PRETRAIN_IMG_SIZE = 224
    cfg.MODEL.SWIN.PATCH_SIZE = 4
    cfg.MODEL.SWIN.EMBED_DIM = 96
    cfg.MODEL.SWIN.DEPTHS = [2, 2, 6, 2]
    cfg.MODEL.SWIN.NUM_HEADS = [3, 6, 12, 24]
    cfg.MODEL.SWIN.WINDOW_SIZE = 7
    cfg.MODEL.SWIN.MLP_RATIO = 4.0
    cfg.MODEL.SWIN.QKV_BIAS = True
    cfg.MODEL.SWIN.QK_SCALE = None
    cfg.MODEL.SWIN.DROP_RATE = 0.0
    cfg.MODEL.SWIN.ATTN_DROP_RATE = 0.0
    cfg.MODEL.SWIN.DROP_PATH_RATE = 0.3
    cfg.MODEL.SWIN.APE = False
    cfg.MODEL.SWIN.PATCH_NORM = True
    cfg.MODEL.SWIN.OUT_FEATURES = ["res2", "res3", "res4", "res5"]

    # NOTE: maskformer2 extra configs
    # transformer module
    cfg.MODEL.MASK_FORMER.TRANSFORMER_DECODER_NAME = "MultiScaleMaskedTransformerDecoder"

    # LSJ aug
    cfg.INPUT.IMAGE_SIZE = 1024
    cfg.INPUT.MIN_SCALE = 0.1
    cfg.INPUT.MAX_SCALE = 2.0

    # MSDeformAttn encoder configs
    cfg.MODEL.SEM_SEG_HEAD.DEFORMABLE_TRANSFORMER_ENCODER_IN_FEATURES = ["res3", "res4", "res5"]
    cfg.MODEL.SEM_SEG_HEAD.DEFORMABLE_TRANSFORMER_ENCODER_N_POINTS = 4
    cfg.MODEL.SEM_SEG_HEAD.DEFORMABLE_TRANSFORMER_ENCODER_N_HEADS = 8

    # point loss configs
    # Number of points sampled during training for a mask point head.
    cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS = 112 * 112
    # Oversampling parameter for PointRend point sampling during training. Parameter `k` in the
    # original paper.
    cfg.MODEL.MASK_FORMER.OVERSAMPLE_RATIO = 3.0
    # Importance sampling parameter for PointRend point sampling during training. Parametr `beta` in
    # the original paper.
    cfg.MODEL.MASK_FORMER.IMPORTANCE_SAMPLE_RATIO = 0.75


def add_our_config(cfg):
    cfg.ORACLE = False
    cfg.PSEUDO = False
    cfg.PSEUDO_WITH_PRIOR = True
    cfg.PSEUDO_REJECT_THRESHOLD = 0.0
    cfg.TEST.SLIDING_WINDOW = False
    cfg.TEST.SLIDING_TILE_SIZE = 224
    cfg.TEST.SLIDING_OVERLAP = 2 / 3.0
    cfg.PSEUDO_FLAG_NAME = "trainable_flag"
    cfg.SOLVER.TEST_IMS_PER_BATCH = 1
    cfg.DATASETS.SAMPLE_PER_CLASS = -1
    cfg.DATASETS.SAMPLE_SEED = 0
    cfg.DATASETS.INPUT_SIZE = [224, 224]
    cfg.DATASETS.FLIP = True
    cfg.DATASETS.SCALE_FACTOR = 0.25
    cfg.DATASETS.ROT_FACTOR = 30
    cfg.DATASETS.NUM_LANDMARKS = 68
    cfg.DATASETS.LANDMARK_INDEX = []
    cfg.DATASETS.BOUNDARY=[]
    cfg.DATASETS.MAX_IMAGE_SIZE = 0
    cfg.DATASETS.LANDMAKR_BOUNDARY_300W=[]
    cfg.DATASETS.LANDMAKR_BOUNDARY_WFLW = []
    cfg.DATASETS.LANDMAKR_BOUNDARY_COFW = []
    cfg.DATASETS.LANDMAKR_BOUNDARY_AFLW = []


    cfg.TEST.OPTIM = CN()
    cfg.TEST.OPTIM.LR = 0.001

    cfg.INPUT.TASK_NAME = ["semantic segmentation."]

    cfg.DATASETS.LANDMARK_INDEX_300W = []
    cfg.DATASETS.LANDMARK_INDEX_COFW = []
    cfg.DATASETS.LANDMARK_INDEX_WFLW = []
    cfg.DATASETS.LANDMARK_INDEX_AFLW = []

    # whether to use dense crf
    cfg.TEST.DENSE_CRF = False
    # embedding head
    cfg.MODEL.SEM_SEG_HEAD.EMBEDDING_DIM = 512
    cfg.MODEL.SEM_SEG_HEAD.EMBED_HIDDEN_DIM = 1024
    cfg.MODEL.SEM_SEG_HEAD.EMBED_LAYERS = 2
    cfg.MODEL.SEM_SEG_HEAD.LAND_CLASSES = 125
    # clip_adapter
    cfg.MODEL.CLIP_ADAPTER = CN()
    cfg.MODEL.CLIP_ADAPTER.PROMPT_LEARNER = "learnable"
    # for predefined
    cfg.MODEL.CLIP_ADAPTER.PREDEFINED_PROMPT_TEMPLATES = ["a sculpture of a {}."]
    # for learnable prompt
    cfg.MODEL.CLIP_ADAPTER.PROMPT_DIM = 512
    cfg.MODEL.CLIP_ADAPTER.PROMPT_SHAPE = (16, 0)
    cfg.MODEL.CLIP_ADAPTER.TASK_PROMPT_SHAPE = 8
    cfg.MODEL.CLIP_ADAPTER.PROMPT_CHECKPOINT = ""
    cfg.MODEL.CLIP_ADAPTER.CLIP_MODEL_NAME = "ViT-B/16"
    cfg.MODEL.CLIP_ADAPTER.MASK_FILL = "mean"
    cfg.MODEL.CLIP_ADAPTER.MASK_EXPAND_RATIO = 1.0
    cfg.MODEL.CLIP_ADAPTER.MASK_THR = 0.5
    cfg.MODEL.CLIP_ADAPTER.MASK_MATTING = False
    cfg.MODEL.CLIP_ADAPTER.REGION_RESIZED = True
    cfg.MODEL.CLIP_ADAPTER.CLIP_ENSEMBLE = True
    cfg.MODEL.CLIP_ADAPTER.CLIP_ENSEMBLE_WEIGHT = 0.8
    #
    cfg.MODEL.CLIP_ADAPTER.SEPERATE_ADAPTER = False
    cfg.MODEL.CLIP_ADAPTER.REGION_CLIP_ADAPTER = CN()
    cfg.MODEL.CLIP_ADAPTER.REGION_CLIP_ADAPTER.CLIP_MODEL_NAME = "ViT-B/16"
    cfg.MODEL.CLIP_ADAPTER.REGION_CLIP_ADAPTER.PROMPT_LEARNER = "predefined"
    # for predefined
    cfg.MODEL.CLIP_ADAPTER.REGION_CLIP_ADAPTER.PREDEFINED_PROMPT_TEMPLATES = [
        "a photo of a {}."
    ]
    # for learnable prompt
    cfg.MODEL.CLIP_ADAPTER.REGION_CLIP_ADAPTER.PROMPT_DIM = 512
    cfg.MODEL.CLIP_ADAPTER.REGION_CLIP_ADAPTER.PROMPT_SHAPE = (16, 0)
    cfg.MODEL.CLIP_ADAPTER.REGION_CLIP_ADAPTER.PROMPT_CHECKPOINT = ""


    cfg.MODEL.SEM_SEG_HEAD.EMB_SIZE = 256
    cfg.MODEL.SEM_SEG_HEAD.EMBED_DIM = 2048
    cfg.MODEL.SEM_SEG_HEAD.NUM_HEADS = 8
    cfg.MODEL.SEM_SEG_HEAD.USE_LAYER_SCALE = True


    # wandb
    cfg.WANDB = CN()
    cfg.WANDB.PROJECT = "zero_shot_seg"
    cfg.WANDB.NAME = None


def add_mask_former_config(cfg):
    """
    Add config for MASK_FORMER.
    """
    add_mask_former_default_config(cfg)
    add_our_config(cfg)
