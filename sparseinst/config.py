# Copyright (c) Tianheng Cheng and its affiliates. All Rights Reserved

from detectron2.config import CfgNode as CN

def add_sparse_inst_config(cfg):

    cfg.MODEL.DEVICE = 'cuda'
    cfg.MODEL.MASK_ON = True
    # [SparseInst]
    cfg.MODEL.SPARSE_INST = CN()


    # parameters for inference
    cfg.MODEL.SPARSE_INST.CLS_THRESHOLD = 0.005
    cfg.MODEL.SPARSE_INST.MASK_THRESHOLD = 0.5
    cfg.MODEL.SPARSE_INST.MAX_DETECTIONS = 100

    # [Encoder]
    cfg.MODEL.SPARSE_INST.ENCODER = CN()
    cfg.MODEL.SPARSE_INST.ENCODER.NAME = "FPNPPMEncoder"
    cfg.MODEL.SPARSE_INST.ENCODER.NORM = ""
    cfg.MODEL.SPARSE_INST.ENCODER.IN_FEATURES = ["res3", "res4", "res5"]
    cfg.MODEL.SPARSE_INST.ENCODER.NUM_CHANNELS = 256

    # [Decoder]
    cfg.MODEL.SPARSE_INST.DECODER = CN()
    cfg.MODEL.SPARSE_INST.DECODER.NAME = "BaseIAMDecoder"
    cfg.MODEL.SPARSE_INST.DECODER.NUM_MASKS = 100
    cfg.MODEL.SPARSE_INST.DECODER.NUM_CLASSES = 1
    # kernels for mask features
    cfg.MODEL.SPARSE_INST.DECODER.KERNEL_DIM = 128
    # upsample factor for output masks
    cfg.MODEL.SPARSE_INST.DECODER.SCALE_FACTOR = 2.0
    cfg.MODEL.SPARSE_INST.DECODER.OUTPUT_IAM = False
    cfg.MODEL.SPARSE_INST.DECODER.GROUPS = 4    


    # decoder.mask_branch
    cfg.MODEL.SPARSE_INST.DECODER.MASK = CN()
    cfg.MODEL.SPARSE_INST.DECODER.MASK.DIM = 256
    cfg.MODEL.SPARSE_INST.DECODER.MASK.CONVS = 3

    # [Loss]
    cfg.MODEL.SPARSE_INST.LOSS = CN()
    cfg.MODEL.SPARSE_INST.LOSS.NAME = "SparseInstCriterion"
    cfg.MODEL.SPARSE_INST.LOSS.ITEMS = ("labels", "masks")

    # generate data
    cfg.MODEL.SPARSE_INST.LOSS.MASK_STRIDE = 8
    cfg.MODEL.SPARSE_INST.LOSS.EMBEDDING_STRIDE = 4
    cfg.MODEL.SPARSE_INST.LOSS.INSTANCE_STRIDE = 4
    cfg.MODEL.SPARSE_INST.LOSS.PULL_PUSH_MARGIN = 0.2
    cfg.MODEL.SPARSE_INST.LOSS.SOFT_MASK_T = 0.9
    cfg.MODEL.SPARSE_INST.LOSS.GAUSS_KERNEL_SIZE = (3, 3)
    cfg.MODEL.SPARSE_INST.LOSS.GAUSS_KERNEL_SIGMA = 7



    # loss weight
    cfg.MODEL.SPARSE_INST.LOSS.CLASS_WEIGHT = 2.0
    cfg.MODEL.SPARSE_INST.LOSS.MASK_PIXEL_WEIGHT = 5.0
    cfg.MODEL.SPARSE_INST.LOSS.MASK_DICE_WEIGHT = 2.0
    # iou-aware objectness loss weight
    cfg.MODEL.SPARSE_INST.LOSS.OBJECTNESS_WEIGHT = 1.0
    cfg.MODEL.SPARSE_INST.LOSS.POS_NUM = 7
    cfg.MODEL.SPARSE_INST.LOSS.TOPK_NUM = 40
    # loss wight
    cfg.MODEL.SPARSE_INST.FOCAL_LOSS_ALPHA = 0.25
    cfg.MODEL.SPARSE_INST.FOCAL_LOSS_GAMMA = 2.0

    # [Matcher]
    cfg.MODEL.SPARSE_INST.MATCHER = CN()
    cfg.MODEL.SPARSE_INST.MATCHER.NAME = "SparseInstMatcher"
    cfg.MODEL.SPARSE_INST.MATCHER.ALPHA = 0.8
    cfg.MODEL.SPARSE_INST.MATCHER.BETA = 0.2

    # [Optimizer]
    cfg.SOLVER.OPTIMIZER = "ADAMW"
    cfg.SOLVER.BACKBONE_MULTIPLIER = 1.0
    cfg.SOLVER.AMSGRAD = False

    # [Dataset mapper]
    cfg.MODEL.SPARSE_INST.DATASET_MAPPER = "SparseInstDatasetMapper"





