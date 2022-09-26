import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from detectron2.config import get_cfg
from detectron2.modeling import build_backbone
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.structures import ImageList, Instances, BitMasks
from detectron2.engine import default_argument_parser, default_setup
from detectron2.data import build_detection_test_loader, build_detection_train_loader
from detectron2.evaluation import COCOEvaluator, print_csv_format
from detectron2.data import  DatasetMapper
from sparseinst import build_sparse_inst_encoder, build_sparse_inst_decoder, add_sparse_inst_config
from sparseinst.gt_generate import GenerateGT
import cv2

device = torch.device('cuda:0')
dtype = torch.float32

__all__ = ["SparseInst"]

pixel_mean = torch.Tensor([123.675, 116.280, 103.530]).to(device).view(3, 1, 1)
pixel_std = torch.Tensor([58.395, 57.120, 57.375]).to(device).view(3, 1, 1)


@torch.jit.script
def normalizer(x, mean, std): return (x - mean) / std


def synchronize():
    torch.cuda.synchronize()


def process_batched_inputs(batched_inputs):
    images = [x["image"].to(device) for x in batched_inputs]
    images = [normalizer(x, pixel_mean, pixel_std) for x in images]
    images = ImageList.from_tensors(images, 32)
    ori_size = (batched_inputs[0]["height"], batched_inputs[0]["width"])
    return images.tensor, images.image_sizes[0], ori_size


@torch.jit.script
def rescoring_mask(scores, mask_pred, masks):
    mask_pred_ = mask_pred.float()
    return scores * ((masks * mask_pred_).sum([1, 2]) / (mask_pred_.sum([1, 2]) + 1e-6))


def cal_similarity(base_w, anchor_w, sim_type="cosine"):
    """
    base_w shape: b, n, dim
    anchor_w shape: b, dim, h ,w

    return: b, n, m
    """
    b, dim, h, w = anchor_w.shape
    _, n, _ = base_w.shape

    anchor_w = anchor_w.reshape(b, dim, -1)

    anchor_w = anchor_w.transpose(-2, -1)  # b, h*w, dim

    if sim_type == "cosine":
        a_n, b_n = base_w.norm(dim=-1).unsqueeze(-1), anchor_w.norm(dim=-1).unsqueeze(-1)

        a_norm = base_w / a_n.clamp(min=1e-8)
        b_norm = anchor_w / b_n.clamp(min=1e-8)
        similarity = torch.matmul(a_norm, b_norm.transpose(-2, -1))
    elif sim_type == "L2":
        similarity = 1. - (base_w - anchor_w).abs().clamp(min=1e-6).norm(dim=1)
    else:
        raise NotImplementedError

    similarity = similarity.reshape(b, n, h, w)

    return similarity



class SparseInst(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.device = torch.device(cfg.MODEL.DEVICE)
        # backbone
        self.backbone = build_backbone(cfg)
        self.size_divisibility = self.backbone.size_divisibility

        output_shape = self.backbone.output_shape()

        self.encoder = build_sparse_inst_encoder(cfg, output_shape)
        self.decoder = build_sparse_inst_decoder(cfg)

        self.to(self.device)

        # inference
        self.cls_threshold = cfg.MODEL.SPARSE_INST.CLS_THRESHOLD
        self.mask_threshold = cfg.MODEL.SPARSE_INST.MASK_THRESHOLD
        self.max_detections = cfg.MODEL.SPARSE_INST.MAX_DETECTIONS
        self.mask_format = cfg.INPUT.MASK_FORMAT
        self.num_classes = cfg.MODEL.SPARSE_INST.DECODER.NUM_CLASSES

    def forward(self, image, resized_size, ori_size):
        max_size = image.shape[2:]
        features = self.backbone(image)
        features = self.encoder(features)
        output = self.decoder(features)
        #result = self.inference_single(output, resized_size, max_size, ori_size)
        return output

    def inference_single(self, outputs, img_shape, pad_shape, ori_shape):
        """
        inference for only one sample
        Args:
            scores (tensor): [NxC]
            masks (tensor): [NxHxW]
            img_shape (list): (h1, w1), image after resized
            pad_shape (list): (h2, w2), padded resized image
            ori_shape (list): (h3, w3), original shape h3*w3 < h1*w1 < h2*w2
        """
        result = Instances(ori_shape)
        # scoring
        pred_logits = outputs["pred_logits"][0].sigmoid().squeeze()
        pred_scores = outputs["pred_masks"][0].sigmoid()


        return result


def preprocess_inputs(batched_inputs):
    images = [x["image"] for x in batched_inputs]

    images = ImageList.from_tensors(images, 32).tensor
    return images



def draw_instance_test(img, instances, instance_indexs):
    for index in range(len(instance_indexs)):
        if instance_indexs[index] == 0:
            break
        mask = instances[index]

        r, g, b = cv2.split(img)
        r[mask > 0.5] = 255

        new_img = cv2.merge([b, g, r])

        cv2.imshow("img", new_img)
        cv2.waitKey(1000)


def draw_guass_map_test(img, guass_map):
    """
    img shape: h, w, 3
    guass_map shape: 1, 1, h, w
    """


    for index in range(len(guass_map)):
        print("img shape", img.shape)


        mask = guass_map[index]

        mask = mask*255
        mask = mask.clip(0, 255)
        mask = mask.astype(np.uint8)

        print(type(mask[0,0]))


        print("mask shape", mask.shape)

        r, g, b = cv2.split(img)
        print(type(b[0, 0]))
        r = mask

        new_img = cv2.merge([b, g, r])

        cv2.imshow("img", new_img)
        cv2.waitKey(1000)



def test_sparseinst_speed(cfg):
    device = torch.device('cuda:0')

    mapper = DatasetMapper(cfg, True)

    data_loader = build_detection_train_loader(cfg, mapper=mapper)

    get_ground_truth = GenerateGT(cfg)



    for idx, inputs in enumerate(data_loader):

        image_tensor = preprocess_inputs(inputs)

        _, _, img_h, img_w = image_tensor.shape

        feat = torch.randn(1, 1, int(img_h/4), int(img_w/4))
        mask_feat = torch.randn(1, 1, int(img_h), int(img_w))

        new_targets = get_ground_truth.generate(inputs, feat, mask_feat)


        gt_scoremaps = new_targets["gt_scoremaps"] #b, class_num, h, w

        print("gt_scoremaps",gt_scoremaps.shape)
        print("image_tensor",image_tensor.shape)

        gt_instances = new_targets["gt_instances"] #b, 100, h, w

        gt_inst_nums = new_targets["gt_inst_nums"].cpu().numpy()  #b, 100

        gt_classes = new_targets["gt_classes"].cpu().numpy()  #b, 100

        gt_scoremaps = F.interpolate(gt_scoremaps, scale_factor=4, mode='bilinear', align_corners=True)

        gt_instances = F.interpolate(gt_instances, scale_factor=4, mode='bilinear', align_corners=True)

        gt_scoremaps = gt_scoremaps.cpu().numpy() #b, class_num, h, w

        gt_instances = gt_instances.cpu().numpy() # b, inst_num, h, w


        imgs = []

        for input in inputs:
            image_tensor = input["image"]

            img = image_tensor.numpy().astype(np.uint8)

            pad_img = np.zeros(shape=(img_h, img_w, 3))

            #print("orgil img shape",img.shape)

            img = img.transpose(1, 2, 0)

            pad_img[:img.shape[0], :img.shape[1], :] = img

            pad_img = pad_img.astype(np.uint8)

            imgs.append(pad_img)



        for index in range(len(imgs)):
            img = imgs[index]
            gt_scoremap = gt_scoremaps[index]
            print("img shape: ", img.shape)
            print("gt_scoremap shape: ", gt_scoremap.shape)
            print("index:", index)

            draw_guass_map_test(img, gt_scoremap)


            gt_instance = gt_instances[index]
            gt_inst_num = gt_inst_nums[index]

            #draw_instance_test(img, gt_instance, gt_inst_num)


            """
            r, g, b = cv2.split(img)
            r[gt_scoremap > 0.5] = 255

            new_img = cv2.merge([b, g, r])

            cv2.imshow("img", new_img)
            cv2.waitKey(1000)
            """














def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_sparse_inst_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


if __name__ == '__main__':
    args = default_argument_parser()
    args = args.parse_args()
    print("Command Line Args:", args)
    cfg = setup(args)
    test_sparseinst_speed(cfg)
