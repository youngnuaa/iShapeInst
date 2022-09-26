import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from detectron2.config import get_cfg
from detectron2.modeling import build_backbone
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.structures import ImageList, Instances, BitMasks
from detectron2.engine import default_argument_parser, default_setup
from detectron2.data import build_detection_test_loader
from detectron2.evaluation import COCOEvaluator, print_csv_format
from sklearn.decomposition import PCA

from sparseinst import build_sparse_inst_encoder, build_sparse_inst_decoder, add_sparse_inst_config
from sparseinst import COCOMaskEvaluator
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

class Gaussian(nn.Module):
    def __init__(self, delta_var=0.2, pmaps_threshold=0.9):
        super().__init__()
        self.delta_var = delta_var
        # dist_var^2 = -2*sigma*ln(pmaps_threshold)
        self.two_sigma = delta_var * delta_var / (-math.log(pmaps_threshold))

    def forward(self, dist_map):
        return torch.exp(- dist_map * dist_map / self.two_sigma)


def cal_2D_similarity(base_w, anchor_w, sim_type="cosine"):
    """
    base_w shape: n, dim
    anchor_w shape: n, dim
    return n, n
    """
    if sim_type == "cosine":
        a_n, b_n = base_w.norm(dim=1).unsqueeze(-1), anchor_w.norm(dim=1).unsqueeze(-1)

        a_norm = base_w / a_n.clamp(min=1e-8)
        b_norm = anchor_w / b_n.clamp(min=1e-8)
        similarity = torch.mm(a_norm, b_norm.transpose(0, 1))
    elif sim_type == "L2":
        similarity = 1. - (base_w - anchor_w).abs().clamp(min=1e-6).norm(dim=1)
    else: raise NotImplementedError
    return similarity


def kernel_fusion(meta_weight, pred_score):

    meta_weight = meta_weight.squeeze(0)
    # meta_weight num, c=64
    if len(meta_weight.shape) == 1:
        meta_weight = meta_weight.unsqueeze(0)

    similarity = cal_2D_similarity(meta_weight, meta_weight)

    label_matrix = similarity.triu(diagonal=0) >= 0.75


    cum_matrix = torch.cumsum(label_matrix.float(), dim=0) < 2

    keep_matrix = cum_matrix.diagonal(0)

    label_matrix = (label_matrix[keep_matrix] & cum_matrix[keep_matrix]).float()

    label_norm = label_matrix.sum(dim=1, keepdim=True)

    #print(label_matrix)

    meta_weight = torch.mm(label_matrix, meta_weight) / label_norm

    pred_score = pred_score[keep_matrix]

    return meta_weight, pred_score


def get_mean_instance(pred_logits, embedding_masks, encode_feat, soft_mask, img = None):
    """
    pred_logits shape: 1, 1, h, w
    soft_mask shape: 1. inst_num, h, w
    embedding_masks shape: 1, dim, h, w
    encode_feat shape:1, dim, 4*h, 4*w
    """
    simi_maps = soft_mask * pred_logits #1. inst_num, h, w

    simi_maps = simi_maps>0.5 #1. inst_num, h, w

    instance_masks = get_mean_embedding(embedding_masks, encode_feat, simi_maps)

    return instance_masks


def cal_feat_similarity(pred_logits, pred_masks):
    """
    pred_logits shape: 1, 1, h, w
    pred_masks shape: 1, feat_dim, h, w

    """

    _, dim, h, w = pred_masks.shape

    pred_logits = pred_logits.squeeze().reshape(-1)

    pred_masks = pred_masks.squeeze()

    pred_logits, index = pred_logits.topk(h*w)

    pred_logits_index = pred_logits>0.75

    index = index[pred_logits_index]
    pred_score = pred_logits[pred_logits_index]


    feat_v = pred_masks.reshape(dim, -1)[:, index].transpose(0, 1) #n, dim.  the score is need sort

    feat_v, pred_score = kernel_fusion(feat_v, pred_score)

    return feat_v, pred_score


def draw_mask(img, pred_logits):
    pred_logits = F.interpolate(pred_logits, scale_factor=4, mode='bilinear', align_corners=True).squeeze()

    pred_logits = pred_logits.cpu().numpy()


    b, g, r = cv2.split(img)

    r[pred_logits > 0.5] = 255

    img2 = cv2.merge([b, g, r])

    cv2.imshow("img", img2)
    cv2.waitKey(1000)


def draw_instance(img, pred_logits, simi_maps):
    pred_logits = F.interpolate(pred_logits, scale_factor=4, mode='bilinear', align_corners=True) #b, 1, h, w
    simi_maps = F.interpolate(simi_maps, scale_factor=4, mode='bilinear', align_corners=True) #b, inst, h, w

    simi_maps = simi_maps*pred_logits

    simi_maps = simi_maps.squeeze()

    simi_maps = simi_maps.cpu().numpy()


    for simi_map in simi_maps:
        b, g, r = cv2.split(img)

        r[simi_map > 0.9] = 255

        img2 = cv2.merge([b, g, r])

        cv2.imshow("img", img2)
        cv2.waitKey(1000)


def draw_embedding_pca_map(instance_map, mask = None, size=None, k=3):

    instance_map = F.interpolate(instance_map, scale_factor=4, mode='bilinear', align_corners=True)

    instance_map = instance_map.squeeze()

    instance_map = instance_map.cpu().numpy()

    instance_map = instance_map.transpose(1,2, 0)

    #print()
    h,w,c = instance_map.shape
    instance_map = instance_map.reshape(-1, c)

    pca = PCA(n_components=k)
    newX = pca.fit_transform(instance_map)

    newX = newX.reshape(h, w, k)

    if mask is not None:
        newX[:, :, 0] = newX[:, :, 0] * mask
        newX[:, :, 1] = newX[:, :, 1] * mask
        newX[:, :, 2] = newX[:, :, 2] * mask
    max_v = np.max(newX)
    min_v = np.min(newX)
    newX = 255*(newX-min_v)/(max_v-min_v)
    newX = np.uint8(newX)
    if size is not None:
        newX = cv2.resize(newX, (size[1], size[0]))

    cv2.imshow("img", newX)
    cv2.waitKey(5000)

    return newX


def get_gt_instance(targets_per_image, feat_shape, device):
    targets_per_image = targets_per_image['instances']

    bit_masks = targets_per_image.gt_masks.tensor.float()

    bit_masks = F.interpolate(bit_masks.unsqueeze(0), size=feat_shape, mode="nearest")
    bit_masks = bit_masks.to(device)

    return bit_masks


def get_mean_embedding_vector(pred_weights, mask_maps):
    """
    pred_weights      :b, num_vector, mask_h, mask_w
    mask_maps  :b, inst_num, mask_h, mask_w

    """
    b, num_vector, mask_h, mask_w = pred_weights.shape

    _, inst_num, _, _ = mask_maps.shape

    gt_inst_piexl_nums = torch.sum(mask_maps.reshape(b, inst_num, -1), dim=-1)

    gt_inst_piexl_nums = torch.clamp(gt_inst_piexl_nums, min=1.0)  # b, inst_num

    weight = pred_weights.unsqueeze(1) * mask_maps.unsqueeze(2)  # b, inst_num, num_vector, mask_h, mask_w

    weight = torch.sum(weight.reshape(b, inst_num, num_vector, -1), dim=-1)  # b, inst_num, num_vector

    mean_embedding_vectors = weight / gt_inst_piexl_nums.unsqueeze(-1)  # b, inst_num, num_vector

    return mean_embedding_vectors, gt_inst_piexl_nums


def get_isntance_mask(embedding_vectors, instance_vector_map):
    """
    embedding_vectors: b,n,embedding_v
    instance_vector_map:b,embedding_v,h, w
    (b,n,embedding_v)*(b,embedding_v,h*w)
    """
    b, c, h, w = instance_vector_map.shape
    #print("instance_vector_map: ", instance_vector_map.shape)
    instance_vector_map = instance_vector_map.reshape(b, c, -1)

    a_n = embedding_vectors.norm(dim=-1).unsqueeze(-1)
    embedding_vectors = embedding_vectors / a_n.clamp(min=1e-8)

    isntance_mask = torch.matmul(embedding_vectors, instance_vector_map)
    #print("isntance_mask", isntance_mask.shape)
    isntance_mask = isntance_mask.reshape(b, -1, h, w)
    return isntance_mask


def get_mean_embedding(embedding_masks, encode_feat, bit_masks, img = None):
    """
    embedding_masks shape: 1, 128, 160, 160
    encode_feat     shape: 1, 128, 640, 640
    bit_masks       shape: 1, inst, 160, 160

    return: mask
    """

    mean_embedding_vectors, gt_inst_mask_pixel_nums = get_mean_embedding_vector(embedding_masks, bit_masks)

    instance_mask = get_isntance_mask(mean_embedding_vectors, encode_feat)

    instance_masks = torch.sigmoid(instance_mask)


    return instance_masks


def get_topk_embedding(embedding_vector, encode_feat):
    """
    embedding_masks shape: 1, 128, 160, 160
    encode_feat     shape: 1, 128, 640, 640
    bit_masks       shape: 1, inst, 160, 160

    return: mask
    """

    instance_mask = get_isntance_mask(embedding_vector, encode_feat)

    instance_masks = torch.sigmoid(instance_mask)
    return instance_masks


def matrix_nms(cate_labels, seg_masks, cate_scores, sigma=2.0, kernel='gaussian'):
    """
    cate_labels  N,1
    seg_masks   N,H,W
    sum_masks   N,1
    cate_scores N,1
    """
    n_samples = len(cate_labels)
    if n_samples == 0:
        return []

    seg_masks = F.interpolate(seg_masks.float().unsqueeze(0), scale_factor=0.5, mode='nearest')[0].bool()

    sum_masks = seg_masks.sum((1, 2)).float()

    seg_masks = seg_masks.reshape(n_samples, -1).float()
    # inter.
    inter_matrix = torch.mm(seg_masks, seg_masks.transpose(1, 0))
    # union.
    sum_masks_x = sum_masks.expand(n_samples, n_samples)
    # iou.
    iou_matrix = (inter_matrix / (sum_masks_x + sum_masks_x.transpose(1, 0) - inter_matrix)).triu(diagonal=1)
    # label_specific matrix.
    cate_labels_x = cate_labels.expand(n_samples, n_samples)
    label_matrix = (cate_labels_x == cate_labels_x.transpose(1, 0)).float().triu(diagonal=1)

    # IoU compensation
    compensate_iou, _ = (iou_matrix * label_matrix).max(0)
    compensate_iou = compensate_iou.expand(n_samples, n_samples).transpose(1, 0)

    # IoU decay / soft nms
    delay_iou = iou_matrix * label_matrix

    # matrix nms
    if kernel == 'linear':
        delay_matrix = (1 - delay_iou) / (1 - compensate_iou)
        delay_coefficient, _ = delay_matrix.min(0)
    else:
        delay_matrix = torch.exp(-1 * sigma * (delay_iou ** 2))
        compensate_matrix = torch.exp(-1 * sigma * (compensate_iou ** 2))
        delay_coefficient, _ = (delay_matrix / compensate_matrix).min(0)

    # update the score.
    cate_scores_update = cate_scores * delay_coefficient

    return cate_scores_update


def get_nms_result(cate_labels, seg_preds, cate_scores, thr):
    seg_masks = seg_preds > thr
    sum_masks = seg_masks.sum((1, 2)).float()
    seg_scores = (seg_preds * seg_masks.float()).sum((1, 2)) / sum_masks
    cate_scores *= seg_scores
    sort_inds = torch.argsort(cate_scores, descending=True)

    seg_masks = seg_masks[sort_inds, :, :]
    seg_preds = seg_preds[sort_inds, :, :]
    cate_scores = cate_scores[sort_inds]
    cate_labels = cate_labels[sort_inds]

    cate_scores = matrix_nms(cate_labels, seg_masks, cate_scores)
    keep = cate_scores >= 0.05

    seg_preds = seg_preds[keep, :, :]
    cate_scores = cate_scores[keep]
    cate_labels = cate_labels[keep]

    return seg_preds, cate_scores, cate_labels

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

        self.soft_mask = Gaussian()

    def forward(self, image, resized_size, ori_size):
        max_size = image.shape[2:]
        features = self.backbone(image)

        l_feat = self.encoder(features)

        output = self.decoder(l_feat)

        result = self.inference_single(output, resized_size, max_size, ori_size)
        return result

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
        pred_center = outputs["pred_logits"].sigmoid()  # h ,w

        embedding_masks = outputs["embedding_feat"]  # h ,w

        encode_feat = outputs["encode_feat"]

        feat_v, pred_score = cal_feat_similarity(pred_center, embedding_masks)

        pred_masks = get_topk_embedding(feat_v, encode_feat)[0]

        labels = torch.zeros_like(pred_score)

        pred_masks, pred_score, labels = get_nms_result(labels, pred_masks, pred_score, self.mask_threshold)

        pred_masks = F.interpolate(pred_masks.unsqueeze(0), size=ori_shape, mode='bilinear', align_corners=False)[0]

        mask_pred = pred_masks > self.mask_threshold

        mask_pred = BitMasks(mask_pred)
        result.pred_masks = mask_pred
        result.scores = pred_score
        result.pred_classes = labels

        return result


def test_sparseinst_speed(cfg):
    device = torch.device('cuda:0')

    model = SparseInst(cfg)
    soft_mask = Gaussian()

    model.eval()
    model.to(device)
    #print(model)
    size = (cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MAX_SIZE_TEST)
    DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
        cfg.MODEL.WEIGHTS, resume=False)

    torch.backends.cudnn.enable = True
    torch.backends.cudnn.benchmark = False

    output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")

    evaluator = COCOMaskEvaluator(cfg.DATASETS.TEST[0], ("segm",), False, output_folder)
    evaluator.reset()
    model.to(device)
    model.eval()
    data_loader = build_detection_test_loader(cfg, cfg.DATASETS.TEST[0])


    durations = []

    with torch.no_grad():
        for idx, inputs in enumerate(data_loader):
            #print(inputs)
            input = inputs[0]
            file_name = input["file_name"]
            image_tensor = input["image"]


            img = cv2.imread(file_name)

            synchronize()
            start_time = time.perf_counter()
            images, resized_size, ori_size = process_batched_inputs(inputs)


            output = model(images, resized_size, ori_size)

            synchronize()
            end = time.perf_counter() - start_time

            durations.append(end)
            if idx % 100 == 0:
                print("process: [{}/{}] fps: {:.3f}".format(idx, len(data_loader), 1/np.mean(durations[100:])))
            evaluator.process(inputs, [{"instances": output}])


    # evaluate
    results = evaluator.evaluate()
    print_csv_format(results)

    latency = np.mean(durations[100:])
    fps = 1 / latency
    print("speed: {:.4f}s FPS: {:.2f}".format(latency, fps))







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
