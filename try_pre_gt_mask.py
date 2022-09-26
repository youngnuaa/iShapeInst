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
from detectron2.data import build_detection_test_loader
from detectron2.structures import Boxes
import math
from sklearn.decomposition import PCA
import random
from sparseinst import build_sparse_inst_encoder, build_sparse_inst_decoder, add_sparse_inst_config
from sparseinst import COCOMaskEvaluator
import cv2
from sparseinst import SparseInstTestDatasetMapper
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


def kernel_fusion(meta_weight, pred_score, index=None):

    meta_weight = meta_weight.squeeze(0)
    # meta_weight num, c=64

    similarity = cal_2D_similarity(meta_weight, meta_weight)

    label_matrix = similarity.triu(diagonal=0) >= 0.85


    cum_matrix = torch.cumsum(label_matrix.float(), dim=0) < 2

    keep_matrix = cum_matrix.diagonal(0)

    label_matrix = (label_matrix[keep_matrix] & cum_matrix[keep_matrix]).float()

    label_norm = label_matrix.sum(dim=1, keepdim=True)

    #print(label_matrix)

    meta_weight = torch.mm(label_matrix, meta_weight) / label_norm

    pred_score = pred_score[keep_matrix]

    if index is not None:
        index = index[keep_matrix]

    return meta_weight, pred_score, index


def cal_feat_similarity(pred_logits, pred_masks):
    """
    pred_logits shape: 1, 1, h, w
    pred_masks shape: 1, feat_dim, h, w

    """

    _, dim, h, w = pred_masks.shape

    pred_logits = pred_logits.squeeze().reshape(-1)

    pred_masks = pred_masks.squeeze()

    pred_logits, index = pred_logits.topk(h*w)

    pred_logits_index = pred_logits > 0.75

    index = index[pred_logits_index]

    #print("index:", index)

    pred_score = pred_logits[pred_logits_index]

    feat_v = pred_masks.reshape(dim, -1)[:, index].transpose(0, 1) #n, dim.  the score is need sort

    feat_v, pred_score, center_index = kernel_fusion(feat_v, pred_score, index)

    #print("index:", center_index)

    return feat_v, pred_score, center_index


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


def random_colors(all_index):
    colors = []
    for index in range(all_index):
        b = random.random()
        g = random.random()
        r = random.random()
        colors.append([int(b * 255), int(g * 255), int(r * 255)])
    colors = np.array(colors)
    return colors


def matrix_nms(cate_labels, seg_masks, sum_masks, cate_scores, sigma=2.0, kernel='gaussian'):
    n_samples = len(cate_labels)
    if n_samples == 0:
        return []

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


def get_nms_result(cate_labels, seg_preds, cate_scores, thr, center_index=None):
    seg_masks = seg_preds > thr
    sum_masks = seg_masks.sum((1, 2)).float()
    seg_scores = (seg_preds * seg_masks.float()).sum((1, 2)) / sum_masks
    cate_scores *= seg_scores
    sort_inds = torch.argsort(cate_scores, descending=True)

    seg_masks = seg_masks[sort_inds, :, :]
    seg_preds = seg_preds[sort_inds, :, :]
    sum_masks = sum_masks[sort_inds]
    cate_scores = cate_scores[sort_inds]
    cate_labels = cate_labels[sort_inds]


    cate_scores = matrix_nms(cate_labels, seg_masks, sum_masks, cate_scores)
    keep = cate_scores >= 0.8

    seg_preds = seg_preds[keep, :, :]
    cate_scores = cate_scores[keep]
    cate_labels = cate_labels[keep]

    if center_index is not None:
        center_index = center_index[sort_inds]
        center_index = center_index[keep]

    return seg_preds, cate_scores, cate_labels, center_index


def get_intersection(pD, pG):
    pInt = pD * pG
    return pInt.sum()


def get_union(pD, pG):
    # print(pD)
    areaA = pD + pG
    areaA = areaA>0
    areaA_sum = areaA.sum()

    return get_intersection(pD, pG)/areaA_sum


def gt_pre_match(result, instances):
    pred_masks = result.pred_masks
    pre_num, _, _ = pred_masks.shape
    gt_masks = instances.gt_masks.tensor.float()
    gt_num, _, _ = gt_masks.shape

    gt_masks = F.interpolate(gt_masks.unsqueeze(0), size=(1024, 1024),
                              mode="bilinear", align_corners=False)[0]

    gt_list_masks = []
    pred_list_masks = []

    for pred_mask in pred_masks:
        pred_list_masks.append(pred_mask.cpu().numpy())

    for gt_mask in gt_masks:
        gt_list_masks.append(gt_mask.numpy())

    draw_p_masks = []
    draw_gt_masks = []

    o_p_masks = []
    o_gt_masks = []

    gt_iou_scores = []
    for gt_ind in range(gt_num):
        gt_mask = gt_list_masks[gt_ind]
        gt_iou_score = []
        for pre_gt in range(pre_num):
            pre_mask = pred_list_masks[pre_gt]
            IOU = get_union(gt_mask, pre_mask)
            gt_iou_score.append(IOU)
        gt_iou_scores.append(gt_iou_score)

    gt_iou_scores = np.array(gt_iou_scores)

    gt_indexs = []
    for gt_ind in range(gt_num):
        gt_mask = gt_list_masks[gt_ind]
        gt_iou_score = gt_iou_scores[gt_ind]
        max_index = np.argmax(gt_iou_score)
        max_score = gt_iou_score[max_index]

        if max_score > 0.5:
            gt_indexs.append(max_index)
            gt_iou_scores[:, max_index] = 0
            draw_gt_masks.append(gt_mask)
            pred_mask = pred_list_masks[max_index]
            draw_p_masks.append(pred_mask)
        else:
            o_gt_masks.append(gt_mask)

    for index in range(pre_num):
        if index in gt_indexs:
            continue
        else:
            o_p_masks.append(pred_list_masks[index])

    return draw_p_masks, o_p_masks, draw_gt_masks, o_gt_masks


def reder_mask(mask_img, pred_mask, color):
    edge_img = draw_instance_edge(pred_mask)
    mask = np.expand_dims(pred_mask, 2)
    mask = mask.repeat(3, axis=2)

    edge_img = np.expand_dims(edge_img, 2)
    edge_img = edge_img.repeat(3, axis=2)

    rgba = np.zeros((pred_mask.shape[0], pred_mask.shape[1], 3))

    rgba[..., 0][pred_mask == 1] = color[0]
    rgba[..., 1][pred_mask == 1] = color[1]
    rgba[..., 2][pred_mask == 1] = color[2]

    mask_img[mask > 0.05] = mask_img[mask > 0.05] * 0.4 + rgba[mask > 0.05] * 0.6
    mask_img[edge_img == 1] = mask_img[edge_img == 1] * 0.0 + rgba[edge_img == 1] * 0
    return mask_img


def write_colors(colors):
    f = open("color.txt", "w")
    for color in colors:
        t_ = str(color[0])+" "+str(color[1])+" "+str(color[2]) + "\n"
        f.writelines(t_)
    f.close()


def read_colors(colors_path):
    f = open(colors_path, "r")
    lines = f.readlines()
    colors = []
    for index in range(len(lines)):
        color = lines[index].replace("\n", "").split(" ")
        for ind in range(3):
            color[ind] = int(color[ind])
        colors.append(color)
    return colors


def draw_img(result, instances, img, img_id):
    """
    """
    labels = result.pred_classes
    if len(labels) == 0:
        print("there are no instance in the img")
        return img

    draw_p_masks, o_p_masks, draw_gt_masks, o_gt_masks = gt_pre_match(result, instances)


    #colors = random_colors(100)
    #write_colors(colors)
    colors = read_colors("/home/data1/data_wy/color.txt")
    print(colors)

    img_path = "./result_img/" + str(img_id) + "_" + str(0) + "ori_img.jpg"
    cv2.imwrite(img_path, img)
    gt_mask_img = img.copy()
    pre_mask_img = img.copy()
    color_ind = 0
    for index in range(len(draw_p_masks)):
        gt_mask = draw_gt_masks[index]
        pre_mask = draw_p_masks[index]
        color = colors[color_ind]

        gt_mask_img = reder_mask(gt_mask_img, gt_mask, color)
        pre_mask_img = reder_mask(pre_mask_img, pre_mask, color)

        gt_img_path = "./result_img/" + str(img_id) + "_" + str(color_ind) + "_gt.jpg"
        cv2.imwrite(gt_img_path, gt_mask_img)

        pre_img_path = "./result_img/" + str(img_id) + "_" + str(color_ind) + "_pre.jpg"
        cv2.imwrite(pre_img_path, pre_mask_img)
        color_ind += 1

    for index in range(max(len(o_p_masks), len(o_gt_masks))):
        color = colors[color_ind]
        if index < len(o_p_masks):
            pre_mask = o_p_masks[index]
            pre_mask_img = reder_mask(pre_mask_img, pre_mask, color)
            pre_img_path = "./result_img/" + str(img_id) + "_" + str(color_ind) + "_pre.jpg"
            cv2.imwrite(pre_img_path, pre_mask_img)
        if index < len(o_gt_masks):
            gt_mask = o_gt_masks[index]
            gt_mask_img = reder_mask(gt_mask_img, gt_mask, color)
            gt_img_path = "./result_img/" + str(img_id) + "_" + str(color_ind) + "_gt.jpg"
            cv2.imwrite(gt_img_path, gt_mask_img)
        color_ind += 1

    return img


def draw_instance_edge(mask):
    h, w = mask.shape
    l_mask = np.zeros(shape=(h+10, w+10))
    l_mask[5:5+h, 5:5+w] = mask

    l_mask = l_mask.astype(np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    e_img = cv2.erode(l_mask, kernel)
    l_mask = l_mask - e_img
    edge_img = l_mask[5:5+h, 5:5+w]
    return edge_img


def gaussian(kernel):
    sigma = ((kernel - 1) * 0.5 - 1) * 0.3 + 0.8
    s = 2 * (sigma ** 2)
    dx = np.exp(-np.square(np.arange(kernel) - int(kernel / 2)) / s)
    return np.reshape(dx, (-1, 1))


def draw_center_point(mask, point, bgr_value=None):
    c_x, c_y = point
    h, w = mask.shape
    p_mask = np.zeros(shape=(h, w))
    p_mask[c_y - 2:c_y + 3, c_x - 2:c_x + 3] = 1.0
    r = 20
    dx = gaussian(r)
    dy = gaussian(r)
    gau_map = np.multiply(dy, np.transpose(dx))


    p_mask[int(c_y - math.floor(r / 2)):int(c_y + math.ceil(r / 2)),
    int(c_x - math.floor(r / 2)):int(c_x + math.ceil(r / 2))] = np.maximum(
        p_mask[int(c_y - math.floor(r / 2)):int(c_y + math.ceil(r / 2)),
        int(c_x - math.floor(r / 2)):int(c_x + math.ceil(r / 2))], gau_map)

    if bgr_value is not None:
        b = p_mask * bgr_value[0]
        g = p_mask * bgr_value[1]
        r = p_mask * bgr_value[2]

    return p_mask


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

        feat_v, pred_score, center_index = cal_feat_similarity(pred_center, embedding_masks)

        pred_masks = get_topk_embedding(feat_v, encode_feat)[0]

        labels = torch.zeros_like(pred_score)

        pred_masks, pred_score, labels, center_index = get_nms_result(labels, pred_masks, pred_score, self.mask_threshold, center_index)

        #print("center_index", center_index)
        pred_masks = F.interpolate(pred_masks.unsqueeze(0), size=ori_shape, mode='bilinear', align_corners=False)[0]

        mask_pred = pred_masks > self.mask_threshold

        #mask_pred = BitMasks(mask_pred)
        result.pred_masks = mask_pred
        result.scores = pred_score
        result.pred_classes = labels
        result.center_index = center_index
        pred_boxes = torch.zeros(mask_pred.size(0), 4)
        result.pred_boxes = Boxes(pred_boxes)
        return result


def test_sparseinst_speed(cfg):
    device = torch.device('cuda:0')

    model = SparseInst(cfg)


    model.eval()
    model.to(device)
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
    maper = SparseInstTestDatasetMapper(cfg, False)
    data_loader = build_detection_test_loader(cfg, cfg.DATASETS.TEST[0], mapper=maper)

    durations = []

    with torch.no_grad():
        for idx, inputs in enumerate(data_loader):
            #print(inputs)
            if idx < 29:
                continue
            input = inputs[0]
            file_name = input["file_name"]
            print(input["file_name"])

            img = cv2.imread(file_name)

            instances = input["instances"]

            synchronize()
            start_time = time.perf_counter()
            [x["image"].to(device) for x in inputs]

            images, resized_size, ori_size = process_batched_inputs(inputs)
            print("images shape", images.shape)
            output = model(images, resized_size, ori_size)

            synchronize()
            end = time.perf_counter() - start_time
            print("start")
            draw_img(output, instances, img, idx)
            ori_img_path = "./result_img/" + str(idx) + "_ori.jpg"
            cv2.imwrite(ori_img_path, img)
            if idx == 29:
                break


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