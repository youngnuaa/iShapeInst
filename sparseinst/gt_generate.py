import numpy as np
import torch
import torch.nn.functional as F
import cv2
from detectron2.structures import ImageList, Instances, BitMasks
from .loss import build_sparse_inst_criterion
from .utils import nested_tensor_from_tensor_list


class GenerateGT(object):
    """
    Generate ground truth for Panoptic FCN.

    """

    def __init__(self, cfg):
        self.device = torch.device(cfg.MODEL.DEVICE)

        self.num_classes = cfg.MODEL.SPARSE_INST.DECODER.NUM_CLASSES
        self.num_masks   = cfg.MODEL.SPARSE_INST.DECODER.NUM_MASKS

        self.mask_stride = cfg.MODEL.SPARSE_INST.LOSS.MASK_STRIDE
        #self.embedding_stride = cfg.MODEL.SPARSE_INST.LOSS.EMBEDDING_STRIDE
        #self.instance_stride = cfg.MODEL.SPARSE_INST.LOSS.INSTANCE_STRIDE

        self.gauss_kernel_size = cfg.MODEL.SPARSE_INST.LOSS.GAUSS_KERNEL_SIZE
        self.gauss_kernel_sigma = cfg.MODEL.SPARSE_INST.LOSS.GAUSS_KERNEL_SIGMA
        self.topk_num = cfg.MODEL.SPARSE_INST.LOSS.TOPK_NUM


    @staticmethod
    def draw_center_map(fmap, mask):
        masked_fmap = torch.max(fmap, mask)
        fmap[:, :] = masked_fmap

    @staticmethod
    def generate_score_map(gt_scoremap, gt_class, bit_masks):
        """
        Generate gaussian-based score map for Things in each stage.
        """
        #print(gt_class.shape)
        #print(bit_masks.shape)

        for i in range(len(gt_class)):
            channel_index = gt_class[i]
            mask = bit_masks[i]
            GenerateGT.draw_center_map(gt_scoremap[channel_index], mask)

    @staticmethod
    def generate_gauss_score_map(gt_scoremap, gt_class, bit_masks, kernel_size, kernel_sigma, device):
        """
        Generate gaussian-based score map for Things in each stage.
        """
        for i in range(len(gt_class)):
            channel_index = gt_class[i]
            mask = bit_masks[i]
            mask = mask.numpy()
            gauss_mask = cv2.GaussianBlur(mask, kernel_size, kernel_sigma)
            gauss_mask = torch.from_numpy(gauss_mask)
            gauss_mask = gauss_mask.to(device)
            GenerateGT.draw_center_map(gt_scoremap[channel_index], gauss_mask)




    def _label_assignment(self, targets_per_image, feat_shape, img_shape):
        #feat_shape is embedding feature and region shape
        #inst_outsize is instance mask output shape
        gt_scoremap = torch.zeros(self.num_classes, *feat_shape, device=self.device)

        gt_instance = torch.zeros(self.num_masks, *feat_shape, device=self.device)

        gt_img_instance = torch.zeros(self.num_masks, *img_shape, device=self.device)

        gt_class = torch.zeros(self.num_masks, device=self.device)

        gt_inst_num = torch.zeros(self.num_masks, device=self.device)

        #gt_range_num = torch.zeros(self.num_masks, self.topk_num, device=self.device)

        range_num = torch.arange(0, self.topk_num)
        range_num = range_num.unsqueeze(0)
        range_num = range_num.expand(self.num_masks, self.topk_num)
        gt_range_num = range_num.to(self.device)

        bit_masks = targets_per_image.gt_masks.tensor.float()

        gt_guass_instance = torch.zeros(len(bit_masks), *feat_shape)

        _, m_h, m_w = bit_masks.shape

        classes = targets_per_image.gt_classes

        num_inst = len(bit_masks)

        gt_inst_num[:num_inst] = 1
        gt_class[:num_inst] = classes.to(self.device)

        gt_img_instance[:num_inst, :bit_masks.shape[1], :bit_masks.shape[2]] = bit_masks.to(self.device)

        #bit_masks = F.interpolate(bit_masks.unsqueeze(0), size=feat_shape, mode="bilinear", align_corners=False)[0]

        bit_masks = F.interpolate(bit_masks.unsqueeze(0), size=(int(m_h/self.mask_stride), int(m_w/self.mask_stride)),
                                  mode="bilinear", align_corners=True)[0]

        #bit_masks[bit_masks>0.5] = 1.0

        #bit_masks[bit_masks < 0.6] = 0.0

        gt_guass_instance[:, :bit_masks.shape[1], :bit_masks.shape[2]] = bit_masks

        gt_instance[:num_inst, :bit_masks.shape[1], :bit_masks.shape[2]] = bit_masks.to(self.device)

        #GenerateGT.generate_gauss_score_map(gt_scoremap, classes, gt_guass_instance, self.gauss_kernel_size, self.gauss_kernel_sigma, self.device)
        GenerateGT.generate_score_map(gt_scoremap, classes, gt_instance)

        gt_instance[gt_instance>0.9] = 1.0

        gt_instance[gt_instance < 0.95] = 0.0
        #gt_scoremap = gt_scoremap

        return gt_scoremap,  gt_instance, gt_inst_num, gt_class, gt_img_instance, gt_range_num

    @torch.no_grad()
    def generate(self, targets, mask_feature, instance_mask):
        """
        Generate ground truth of multi-stages according to the input.
        """

        _, _, feat_h, feat_w = mask_feature.shape


        _, _, mask_h, mask_w = instance_mask.shape


        new_targets = {}

        gt_scoremaps, gt_instances, inst_nums, gt_classes,  gt_inst_pixel_nums= [], [], [], [], []

        gt_img_instances = []

        gt_range_nums = []

        for targets_per_image in targets:
            #img = targets_per_image["image"]

            #print("img shape", img.shape)

            targets_per_image = targets_per_image['instances']

            gt_scoremap, gt_instance, gt_inst_num, gt_class, gt_img_instance, gt_range_num = \
                self._label_assignment(targets_per_image, (feat_h, feat_w), (mask_h, mask_w))
            #print("gt_scoremap shape", gt_scoremap.shape)
            gt_scoremaps.append(gt_scoremap)
            gt_instances.append(gt_instance)
            inst_nums.append(gt_inst_num)
            gt_classes.append(gt_class)
            gt_img_instances.append(gt_img_instance)
            gt_range_nums.append(gt_range_num)


        gt_scoremaps = torch.stack(gt_scoremaps, dim=0)
        gt_instances = torch.stack(gt_instances, dim=0)
        inst_nums = torch.stack(inst_nums, dim=0)
        gt_classes = torch.stack(gt_classes, dim=0)
        gt_img_instances = torch.stack(gt_img_instances, dim=0)

        gt_range_nums = torch.stack(gt_range_nums, dim=0)

        new_targets["gt_scoremaps"] = gt_scoremaps

        new_targets["gt_instances"] = gt_instances

        new_targets["gt_inst_nums"] = inst_nums

        new_targets["gt_classes"] = gt_classes

        new_targets["gt_img_instances"] = gt_img_instances

        new_targets["gt_range_nums"] = gt_range_nums

        return new_targets