# Copyright (c) Tianheng Cheng and its affiliates. All Rights Reserved

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from fvcore.nn import sigmoid_focal_loss_jit
import math
from detectron2.utils.registry import Registry
from .utils import nested_masks_from_list, is_dist_avail_and_initialized, get_world_size


SPARSE_INST_MATCHER_REGISTRY = Registry("SPARSE_INST_MATCHER")
SPARSE_INST_MATCHER_REGISTRY.__doc__ = "Matcher for SparseInst"
SPARSE_INST_CRITERION_REGISTRY = Registry("SPARSE_INST_CRITERION")
SPARSE_INST_CRITERION_REGISTRY.__doc__ = "Criterion for SparseInst"


class Gaussian(nn.Module):
    def __init__(self, delta_var=0.5, pmaps_threshold=0.9):
        super().__init__()
        self.delta_var = delta_var
        # dist_var^2 = -2*sigma*ln(pmaps_threshold)
        self.two_sigma = delta_var * delta_var / (-math.log(pmaps_threshold))

    def forward(self, dist_map):
        return torch.exp(- dist_map * dist_map / self.two_sigma)


class PushLoss(nn.Module):
    def __init__(self, margin=0.2, reduction="sum"):
        super().__init__()
        self.margin = margin
        self.reduction = reduction


    def forward(self, mean_embedding_vector, index_mask):
        """
        mean_embedding_vector: b, inst_num, v_num
        gt_index_mask shape: b, inst_num
        """
        b, inst_num, v_c = mean_embedding_vector.shape
        matrix1 = mean_embedding_vector.unsqueeze(2).expand(b, inst_num, inst_num, v_c)
        matrix2 = mean_embedding_vector.unsqueeze(1).expand(b, inst_num, inst_num, v_c)
        dist_matrix = torch.norm(matrix1 - matrix2, dim=-1) #b, inst_num, inst_num

        neg_index = index_mask.unsqueeze(1).expand(b, inst_num, inst_num)

        pos_index = index_mask.unsqueeze(2).expand(b, inst_num, inst_num)

        mask = neg_index * pos_index

        mask = mask.triu(1)  #b, inst, inst

        index = torch.nonzero(mask)

        dist_matrix = dist_matrix[index[:, 0], index[:, 1], index[:, 2]]

        loss = torch.clamp(2*self.margin - dist_matrix, min=0) ** 2

        if self.reduction == "sum":
            loss = loss.sum()
        elif self.reduction == "mean":
            loss = loss.mean()

        return loss


class PullLosee(nn.Module):
    def __init__(self, margin=0.5, reduction='sum'):
        super().__init__()
        self.margin = margin
        self.reduction = reduction


    def forward(self, mean_embedding_vector, embedding_map, gt_instance, gt_index_mask):
        """
        mean_embedding_vector: b, inst_num, v_num
        embedding_map: b, v_num, h, w
        gt_instance: b, inst_num, h, w
        """
        b, inst_num, v_dim = mean_embedding_vector.shape
        _, _, h, w = embedding_map.shape

        index_mask = gt_index_mask.reshape(-1).bool()

        gt_instance = gt_instance.reshape(b, inst_num, -1)

        inst_pixel_nums = gt_instance.sum(dim=-1)

        inst_pixel_nums = inst_pixel_nums.reshape(-1)[index_mask]  # b*inst_num

        embedding_map = embedding_map.reshape(b, v_dim, -1).transpose(2, 1) #b, h*w, v_dim

        diff = mean_embedding_vector.unsqueeze(2)  - embedding_map.unsqueeze(1) #b, inst_num, h*w, v_dim

        dist_matrix = torch.norm(diff,  dim=-1) ##b, inst_num, h*w

        dist_matrix = dist_matrix*gt_instance

        dist_matrix = torch.clamp(dist_matrix - self.margin, min=0)

        dist_to_mean = dist_matrix ** 2  #b, inst_num, h*w

        dist_to_mean = dist_to_mean.reshape(-1, h*w)[index_mask, ...] #_, h*w

        dist_to_mean = dist_to_mean.sum(dim=-1)/inst_pixel_nums

        if self.reduction == "sum":
            loss = torch.sum(dist_to_mean)
        else:
            loss = torch.mean(dist_to_mean)


        return loss


class EuSoftMaskLoss(nn.Module):
    def __init__(self, delta_var=0.2, pmaps_threshold=0.9, reduction='sum', pos_num=None):
        super().__init__()
        self.dist_to_mask = self.EuGaussian(delta_var, pmaps_threshold)

        self.reduction = reduction

        self.pose_num = pos_num


    def forward(self, pre_masks, gt_instances, index_mask):
        """
        pre_mask is cos map. the shape b, inst_pos_num, h, w
        index_mask shape: b, inst_num

        """
        b, inst_pos_num, mask_h, mask_w = pre_masks.shape
        _, inst_num, _, _ = gt_instances.shape

        if self.pose_num:
            gt_instances = gt_instances.unsqueeze(2).expand(b, inst_num, self.pose_num, mask_h, mask_w)
            index_mask = index_mask.unsqueeze(-1).expand(b, inst_num, self.pose_num)


        index_mask = index_mask.reshape(-1).bool()

        pre_masks = pre_masks.reshape(-1, mask_h, mask_w)[index_mask, ...]

        gt_instances = gt_instances.reshape(-1, mask_h, mask_w)[index_mask, ...]

        pre_masks = self.dist_to_mask(pre_masks)

        #loss = self.ohem_loss(pre_masks, gt_instances)
        loss = w_dice_loss(pre_masks, gt_instances, reduction="sum")

        if self.reduction == 'sum':
            loss = loss.sum()
        else:
            loss = loss.mean()

        return loss



    class EuGaussian(nn.Module):
        def __init__(self, delta_var, pmaps_threshold):
            super().__init__()
            self.delta_var = delta_var
            # dist_var^2 = -2*sigma*ln(pmaps_threshold)
            self.two_sigma = delta_var * delta_var / (-math.log(pmaps_threshold))

        def forward(self, dist_map):
            return torch.exp(- dist_map * dist_map / self.two_sigma)




class SoftMaskLoss(nn.Module):
    def __init__(self, delta_var=0.3, pmaps_threshold=0.9, reduction='sum', pos_num = None):
        super().__init__()
        self.dist_to_mask = Gaussian(delta_var, pmaps_threshold)
        #self.ohem_loss = OhemCELoss(0.7)
        self.reduction = reduction
        self.pos_num = pos_num

    def forward(self, pre_masks, gt_instances, index_mask, weighted_val=None):
        """
        pre_mask is cos map. b, inst_num*7, h, w
        index_mask shape: b, inst_num
        weight_value shape: b, inst_num*7
        """
        b, inst_pos_num, mask_h, mask_w = pre_masks.shape
        _, inst_num, _, _ = gt_instances.shape


        index_mask = index_mask.reshape(-1).bool()

        num = max(index_mask.reshape(-1).sum(), 1)*self.pos_num
        # print(pre_masks.shape)
        # print(gt_instances.shape)
        pre_masks = pre_masks.reshape(-1, self.pos_num, mask_h, mask_w)[index_mask, ...]
        gt_instances = gt_instances.reshape(-1, mask_h, mask_w)[index_mask, ...]
        gt_instances = gt_instances.unsqueeze(1).expand(gt_instances.shape[0], self.pos_num, mask_h, mask_w)

        if weighted_val is not None:
            weighted_val = weighted_val.reshape(-1, self.pos_num)[index_mask, ...]  # x, pose

        pre_masks = self.dist_to_mask(1 - pre_masks)
        # loss = self.ohem_loss(pre_masks, gt_instances)
        # print("index_mask", index_mask.sum())
        # print("pre_masks", pre_masks.shape)
        # print("gt_instances", gt_instances.shape)
        loss = self.w_dice_loss(pre_masks, gt_instances, weighted_val, reduction=self.reduction)/num
        return loss


    def w_dice_loss(self, prediction, target_seg, weighted_val=None, reduction: str = "sum", eps: float = 1e-8):
        # calculate dice loss
        loss_part = (prediction ** 2).sum(dim=-1) + (target_seg ** 2).sum(dim=-1)
        loss = 1 - 2 * (target_seg * prediction).sum(dim=-1) / torch.clamp(loss_part, min=eps)

        if weighted_val is not None:
            weighted_val = weighted_val.unsqueeze(-1)
            loss = loss * weighted_val

        if reduction == "sum":
            loss = loss.sum()
        elif reduction == "mean":
            #print(loss.shape)
            loss = loss.mean()
        return loss



def compute_mask_iou(inputs, targets):
    inputs = inputs.sigmoid()
    # thresholding 
    binarized_inputs = (inputs >= 0.4).float()
    targets = (targets > 0.5).float()
    intersection = (binarized_inputs * targets).sum(-1)
    union = targets.sum(-1) + binarized_inputs.sum(-1) - intersection
    score = intersection / (union + 1e-6)
    return score


def dice_score(inputs, targets):
    numerator = 2 * torch.matmul(inputs, targets.t())
    denominator = (inputs * inputs).sum(-1)[:, None] + (targets * targets).sum(-1)
    score = numerator / (denominator + 1e-4)
    return score


def dice_loss(inputs, targets, reduction='sum'):
    inputs = inputs.sigmoid()
    assert inputs.shape == targets.shape
    numerator = 2 * (inputs * targets).sum(1)
    denominator = (inputs * inputs).sum(-1) + (targets * targets).sum(-1)
    loss = 1 - (numerator) / (denominator + 1e-4)
    if reduction == 'none':
        return loss
    return loss.sum()


def cal_similarity(base_w, anchor_w, sim_type="cosine"):
    if sim_type == "cosine":
        a_n, b_n = base_w.norm(dim=1).unsqueeze(-1), anchor_w.norm(dim=1).unsqueeze(-1)
        a_norm = base_w / a_n.clamp(min=1e-8)
        b_norm = anchor_w / b_n.clamp(min=1e-8)
        similarity = torch.mm(a_norm, b_norm.transpose(0, 1))
    elif sim_type == "L2":
        similarity = 1. - (base_w - anchor_w).abs().clamp(min=1e-6).norm(dim=1)
    else: raise NotImplementedError
    return similarity



def embedding_loss(inputs, inst_indexs, inst_labels, inst_nums):
    """
    inputs shape =  b, embedding_feat_c, h, w
    inst_indexs(list) = inst_num
    inst_labels(list) = inst_num
    inst_nums = b, 100
    """
    inst_nums = torch.sum(inst_nums, dim=-1)
    inst_nums = torch.clamp(inst_nums, 1)
    b, embedding_feat_c, h, w = inputs.shape
    inputs = inputs.reshape(b, embedding_feat_c, -1)
    inputs = inputs.transpose(2, 1) #b, h*w, embedding_feat_c

    losses = []
    #split merge  cal loss
    for index in range(b):
        input = inputs[index]  #h*w, embedding_feat_c
        inst_index = inst_indexs[index]  #num
        inst_label = inst_labels[index]

        embendding_feat = torch.index_select(input, 0, inst_index)

        similarity = cal_similarity(embendding_feat, embendding_feat)


        #there are some problem. if similarity is 0,then value is miss?
        similarity = torch.triu(similarity, diagonal=1)
        embedding_label = similarity[similarity > 0]/2 + 0.5

        ce_loss = F.binary_cross_entropy_with_logits(embedding_label, inst_label, reduction="none")

        loss = ce_loss.sum()
        losses.append(loss)

    losses = torch.stack(losses,dim=0)/inst_nums  #this need multiy instane num
    losses = torch.sum(losses)
    return losses


def get_isntance_mask(embedding_vectors, instance_vector_map):
    """
    embedding_vectors: b,n,embedding_v
    instance_vector_map:b,embedding_v,h, w
    (b,n,embedding_v)*(b,embedding_v,h*w)
    """
    b, c, h, w = instance_vector_map.shape
    instance_vector_map = instance_vector_map.reshape(b, c, -1)
    isntance_mask = torch.matmul(embedding_vectors, instance_vector_map)
    isntance_mask = isntance_mask.reshape(b, -1, h, w)
    return isntance_mask


#hight score to predict instance mask
def cal_high_score_instance_mask(predict_mask, predict_embedding_vector,  gt_embeddingmaps):
    """

    """
    batch_num, c_num, h, w = predict_mask.shape
    _, v_num, _, _ = predict_embedding_vector.shape

    pre_regions = predict_mask*gt_embeddingmaps  #batch_num, inst_num, h, w

    weighted_values, guided_index = torch.topk(pre_regions.reshape(*pre_regions.shape[:2], -1),
                                               k=7, dim=-1)   #b, 100, 7


    guided_index = guided_index.reshape(batch_num, -1) #b, thing_num*7

    inst_w = predict_embedding_vector.reshape(batch_num, v_num, -1) # batch_num, v_num, h*w

    idx_inst = guided_index.unsqueeze(1).expand(batch_num, v_num, -1)#b,1,thing_num*7 --- batch_num, v_num, thing_num*7
    idx_feat_th = torch.gather(inst_w, dim=2, index=idx_inst)  #batch_num, v_num, thing_num*7
    idx_feat_th = idx_feat_th.permute(0, 2, 1)

    return idx_feat_th, weighted_values


def get_mean_embedding_vector(pred_weights, mask_maps):
    """
    pred_weights      :b, num_vector, mask_h, mask_w
    gt_embeddingmaps  :b, inst_num, mask_h, mask_w
    gt_inst_mask_nums :b, inst_num
    """
    b, num_vector, mask_h, mask_w = pred_weights.shape
    _, inst_num, _, _ = mask_maps.shape


    gt_inst_piexl_nums = torch.sum(mask_maps.reshape(b, inst_num, -1), dim=-1)
    gt_inst_piexl_nums = torch.clamp(gt_inst_piexl_nums, min=1.0)

    weight = pred_weights.unsqueeze(1)*mask_maps.unsqueeze(2)  #b, inst_num, num_vector, mask_h, mask_w
    weight = torch.sum(weight.reshape(b, inst_num, num_vector, -1), dim=-1) #b, inst_num, num_vector

    mean_embedding_vectors = weight/gt_inst_piexl_nums.unsqueeze(-1) #b, inst_num, num_vector

    return mean_embedding_vectors, gt_inst_piexl_nums


def get_gt_pre_inst_mask(isntance_masks, gt_instances, gt_index_mask, weighted_num,  weighted_val=None, eps=1e-8):
    """
    isntance_masks: batch_num, thing_num*7, H, W
    gt_instances:   batch_num, thing_num, H, W
    gt_inst_nums:   batch_num, thing_num

    """
    index_mask = gt_index_mask.reshape(-1).bool()
    gt_num = torch.sum(index_mask)
    batch_num, all_inst_num, mask_h, mask_w = isntance_masks.shape
    _, inst_num, _, _ = gt_instances.shape
    prediction = isntance_masks.reshape(batch_num, inst_num, weighted_num, mask_h, mask_w)
    prediction = prediction.reshape(-1, weighted_num, mask_h, mask_w)[index_mask, ...]

    target_seg = gt_instances.unsqueeze(2).expand(batch_num, inst_num, weighted_num, mask_h, mask_w)
    target_seg = target_seg.reshape(-1, weighted_num, mask_h, mask_w)[index_mask, ...]

    if weighted_val:
        weighted_val = weighted_val.reshape(-1, weighted_num)[index_mask, ...]
        weighted_val = weighted_val / torch.clamp(weighted_val.sum(dim=-1, keepdim=True), min=eps)

    prediction = torch.sigmoid(prediction)
    prediction = prediction.reshape(int(gt_num), weighted_num, mask_h * mask_w)
    target_seg = target_seg.reshape(int(gt_num), weighted_num, mask_h * mask_w)

    return prediction, target_seg, weighted_val


def get_gt_mean_inst_mask(mean_isntance_masks, gt_instances, gt_index_mask):
    """
    isntance_masks: batch_num, thing_num, H, W
    gt_instances:   batch_num, thing_num, H, W
    gt_inst_nums:   batch_num, thing_num
    """

    index_mask = gt_index_mask.reshape(-1).bool()

    batch_num, inst_num, mask_h, mask_w = mean_isntance_masks.shape
    _, inst_num, _, _ = gt_instances.shape

    prediction = mean_isntance_masks.reshape(-1, mask_h, mask_w)[index_mask, ...]
    target_seg = gt_instances.reshape(-1, mask_h, mask_w)[index_mask, ...]

    prediction = torch.sigmoid(prediction)
    prediction = prediction.reshape(-1, mask_h * mask_w)
    target_seg = target_seg.reshape(-1, mask_h * mask_w)

    return prediction, target_seg


def loss(pred_mask, encode_feat, gt_dict):
    """
    Args:
        pred_mask shape: b, num_class, mask_h, mask_w
        pred_weights shape: b, num_vector, mask_h, mask_w
        encode_feat shape: b, num_vector, feat_h, feat_w
        gt_dict = {
            "gt_scoremaps": mask segmentation, :b, num_class, feat_h, feat_w
            "gt_instances": instance mask,     :b, inst_num, feat_h, feat_w
            "gt_inst_nums": instance num,      :b, 100
            "gt_classes": instance classes,    :b, 100
        }

    Returns:
        loss(dict): a dict contains all information of loss function
        loss = {
            "loss_mask": position loss for things,
            "loss_embedding_mean": segmentation loss for things,
            "loss_instance": segmentation loss for things,
        }
    """
    gt_scoremaps = gt_dict["gt_scoremaps"]
    gt_instances = gt_dict["gt_instances"]
    gt_index_mask = gt_dict["gt_inst_nums"]
    #gt_classes = gt_dict["gt_classes"]

    thing_num = int(max(gt_index_mask.sum(dim=1).max(), 1))

    gt_index_mask = gt_index_mask[:, :thing_num]
    gt_instances = gt_instances[:, :thing_num]
    #gt_classes = gt_classes[:, :thing_num]

    mean_embedding_vectors, gt_inst_mask_pixel_nums = get_mean_embedding_vector(encode_feat, gt_instances)

    idx_feat_th, weighted_values = cal_high_score_instance_mask(pred_mask, encode_feat, gt_instances)

    isntance_masks = get_isntance_mask(idx_feat_th, encode_feat)   #batch_num, thing_num*7, H, W

    mean_isntance_masks = get_isntance_mask(mean_embedding_vectors, encode_feat)

    mean_prediction, mean_target_seg = get_gt_mean_inst_mask(mean_isntance_masks, gt_instances, gt_index_mask)

    embedding_l2_loss = l2_loss(encode_feat, mean_embedding_vectors, gt_instances, gt_inst_mask_pixel_nums)

    prediction, target_seg, weighted_val = get_gt_pre_inst_mask(isntance_masks, gt_instances, gt_index_mask, 7)

    inst_loss = w_dice_loss(prediction, target_seg, weighted_val)

    mean_inst_loss = w_dice_loss(mean_prediction, mean_target_seg)


def cosine_loss(pre_mask, gt_mask, reduction = "sum"):
    """
    pre_mask, n, h ,w
    (-1, 1)
    """

    pos_loss = -torch.log((pre_mask+1)/2)


    #neg_value = pre_mask.clamp(min=0)

    net_loss = -torch.log(1-pre_mask.clamp(min=0.0))

    loss = pos_loss*gt_mask + net_loss*(1-gt_mask)

    loss = loss.reshape(loss.shape[0], -1)
    loss = loss.mean(-1)

    if reduction == "sum":
        loss = loss.sum()
    else:
        loss = loss.mean()


    return loss


#mean embedding vector to predict instance mask
def cal_mean_embedding_vector_instance_mask():
    mask = 0
    return mask


class PullPushLoss(nn.Module):
    """
    there is some problems, the pull and push margin is same, we can set different marion for pos and neg.!!!!
    """

    def __init__(self, ohem_num = None, margin=0.2, pos_num = None):
        super(PullPushLoss, self).__init__()

        self.ohem_num = ohem_num
        self.margin = margin
        self.pos_num = pos_num


    def forward(self, cos_maps, gt_instance, index_mask):
        b_size, inst_pos_num, h, w = cos_maps.shape
        _, inst_num, _, _ = gt_instance.shape
        if self.pos_num:
            index_mask = index_mask.unsqueeze(-1).expand(b_size, inst_num, self.pos_num).transpose(2, 1)
            gt_instance = gt_instance.unsqueeze(2).expand(b_size, inst_num, self.pos_num, h, w).transpose(2, 1)
            index_mask = index_mask.reshape(-1, inst_num)
            gt_instance = gt_instance.reshape(-1, inst_num, h, w)
            cos_maps = cos_maps.reshape(b_size, inst_num, -1, h, w).transpose(2, 1).reshape(-1, inst_num, h, w)

            b_size, _, _, _ = cos_maps.shape


        neg_index = index_mask.unsqueeze(1).expand(b_size, inst_num, inst_num)

        pos_index = index_mask.unsqueeze(2).expand(b_size, inst_num, inst_num)

        mask = neg_index * pos_index

        mask = mask.triu(1)

        index = torch.nonzero(mask)

        neg_maps = cos_maps.unsqueeze(1).expand(b_size, inst_num, inst_num, h, w)

        pos_maps = cos_maps.unsqueeze(2).expand(b_size, inst_num, inst_num, h, w)

        dis = neg_maps - pos_maps + self.margin

        dis = dis.clamp(min=0.0)

        instance_masks = gt_instance.unsqueeze(2).expand(b_size, inst_num, inst_num, h, w)

        instance_masks = instance_masks[index[:, 0], index[:, 1], index[:, 2], :, :]

        loss = dis[index[:, 0], index[:, 1], index[:, 2], :, :] * instance_masks

        loss = -torch.log(1 - loss / (2 + self.margin))  # b, inst, inst, h, w

        if self.ohem_num:
            loss = loss.reshape(loss.shape[0], -1)

            loss, _ = loss.topk(self.ohem_num, dim=-1)

        loss = loss.sum()

        return loss



def pull_push_loss(cos_maps, gt_instance, index_mask, ohem_num = None, margin=0.3):
    """
    cos_maps shape: b, n, h, w
    gt_instance shape: b, n, w, h
    example:
    [1,2,3,4]
    pos ={
        1 1 1 1
        2 2 2 2
        3 3 3 3
        4 4 4 4
        }

    neg ={
        1 2 3 4
        1 2 3 4
        1 2 3 4
        1 2 3 4
    }

    return:  loss

    #how to set ohem.

    """


    b_size, inst_num, h ,w = cos_maps.shape

    neg_index = index_mask.unsqueeze(1).expand(b_size, inst_num, inst_num)

    pos_index = index_mask.unsqueeze(2).expand(b_size, inst_num, inst_num)

    mask = neg_index*pos_index

    mask = mask.triu(1)

    index = torch.nonzero(mask)

    neg_maps = cos_maps.unsqueeze(1).expand(b_size, inst_num, inst_num, h ,w)

    pos_maps = cos_maps.unsqueeze(2).expand(b_size, inst_num, inst_num, h ,w)

    dis = neg_maps - pos_maps + margin

    dis = dis.clamp(min=0.0)


    instance_masks = gt_instance.unsqueeze(2).expand(b_size, inst_num, inst_num, h ,w)

    instance_masks = instance_masks[index[:,0], index[:,1], index[:,2], :, :]

    loss = dis[index[:,0], index[:,1], index[:,2], :, :]*instance_masks

    loss = -torch.log(1-loss/(2 + margin))  #b, inst, inst, h, w

    if ohem_num:
        loss = loss.reshape(loss.shape[0], -1)

        loss, _ = loss.topk(ohem_num, dim=-1)

    loss = loss.sum()

    return loss


class L2Loss(nn.Module):
    """
    there is some problems, the pull and push margin is same, we can set different marion for pos and neg.!!!!
    """

    def __init__(self, pos_num=None, reduction="sum"):
        super(L2Loss, self).__init__()
        self.reduction = reduction
        self.pos_num = pos_num


    def forward(self, idx_feat_th, embedding_feat, gt_instance, gt_index_mask):
        """
        idx_feat_th shape: b, inst_num*pos_num, v_num
        embedding_feat shape: b, v_num, h, w
        gt_instance shape:  b, inst_num, h, w
        gt_index_mask shape: b, inst_num
        """

        b, inst_pos_num, v_num = idx_feat_th.shape

        _, inst_num, h, w = gt_instance.shape



        gt_instance = gt_instance.unsqueeze(2).expand(b, inst_num, self.pos_num, h, w).reshape(b, inst_num*self.pos_num, -1)

        gt_instance_pixel_num = gt_instance.sum(dim=-1).clamp(min=1.0)

        gt_index_mask = gt_index_mask.unsqueeze(-1).expand(b, inst_num, self.pos_num).reshape(b, -1)


        index_mask = gt_index_mask.reshape(-1).bool()

        inst_pixel_nums = gt_instance_pixel_num.reshape(-1)[index_mask]  # b*inst_pos_num


        inputs = embedding_feat.reshape(b, v_num, -1)

        diff = inputs.unsqueeze(1) - idx_feat_th.unsqueeze(-1)  # b, inst_pos_num, v_num, h*w

        diff = (diff * gt_instance.unsqueeze(-2)) ** 2  # b, inst_pos_num, v_num, h*w

        diff = diff.reshape(-1, v_num, h * w)[index_mask, ...]  # b*inst_pos_num, v_num, h*w

        diff = torch.sum(diff, dim=-1)  # b*inst_pos_num, v_num

        diff = diff / inst_pixel_nums.unsqueeze(-1)  # b*inst_pos_num, v_num

        loss = torch.sum(diff, dim=-1)  # b*inst_pos_num

        if self.reduction == "sum":
            loss = loss.sum()
        elif self.reduction == "mean":
            loss = loss.mean()

        return loss


def l2_loss(inputs, targets, mask, inst_pixel_nums, gt_index_mask, reduction = "sum"):
    """
    inputs shape =  b, embedding_feat_c, h, w
    targets shape = b, inst_num, embedding_feat_c // mean embedding vector of each instance
    mask shape = b, inst_num, h, w
    inst_pixel_nums shape = b, inst_num
    gt_index_mask = b, inst_num
    """
    b, inst_num, embedding_feat_c = targets.shape

    _, _, h, w = inputs.shape

    index_mask = gt_index_mask.reshape(-1).bool()

    inst_pixel_nums = inst_pixel_nums.reshape(-1)[index_mask]  #b*inst_num

    mask = mask.reshape(b, inst_num, -1) #b, inst_num, h*w

    inputs = inputs.reshape(b, embedding_feat_c, -1)

    diff = inputs.unsqueeze(1)- targets.unsqueeze(-1)  #b, inst_num, embedding_feat_c, h*w

    diff = (diff*mask.unsqueeze(-2))**2  #b, inst_num, embedding_feat_c, h*w

    diff = diff.reshape(-1, embedding_feat_c, h*w)[index_mask, ...]  #b*inst_num, embedding_feat_c, h*w

    diff = torch.sum(diff, dim=-1) #b*inst_num, embedding_feat_c
    diff = diff/inst_pixel_nums.unsqueeze(-1) #b*inst_num, embedding_feat_c

    loss = torch.sum(diff, dim=-1)  #b*inst_num

    if reduction == "sum":
        loss = loss.sum()
    elif reduction == "mean":
        loss = loss.mean()

    return loss


def sigmoid_focal_loss(
    inputs,
    targets,
    mode: str = "thing",
    alpha: float = -1,
    gamma: float = 2,
    reduction: str = "none",
):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        mode: A string used to indicte the optimization mode.
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
        reduction: 'none' | 'mean' | 'sum'
                 'none': No reduction will be applied to the output.
                 'mean': The output will be averaged.
                 'sum': The output will be summed.
    Returns:
        Loss tensor with the reduction option applied.
    """
    p = torch.sigmoid(inputs)
    ce_loss = F.binary_cross_entropy_with_logits(
        inputs, targets, reduction="none"
    )
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    # pixel-wise loss for stuff
    if mode == "stuff":
        loss = loss.reshape(*loss.shape[:2], -1).mean(dim=-1)

    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()

    return loss


class OhemCELoss(nn.Module):

    def __init__(self, thresh):
        super(OhemCELoss, self).__init__()
        self.thresh = -torch.log(torch.tensor(thresh, requires_grad=False, dtype=torch.float)).cuda()
        self.criteria = nn.BCELoss(reduction="none")


    def forward(self, logits, labels):
        """
        :param logits: n,3,h,w
        :param labels: n,3,h,w
        :param mask:
        :return:
        """
        #logits = torch.sigmoid(logits)
        shape = logits.shape
        if len(shape) == 2:
            n_min = logits.shape[0]*logits.shape[1] // 16
        elif len(shape) == 3:
            n_min = logits.shape[0] * logits.shape[1] * logits.shape[2] // 16
        else:
            n_min = logits.shape[0] * logits.shape[1] * logits.shape[2] * logits.shape[3] // 16

        loss = (self.criteria(logits, labels)).reshape(-1)
        try:
            loss_hard = loss[loss > self.thresh]
        except:
            print("error seg loss ###########################",loss.shape)
            loss_hard = logits.sum() * 0
            return loss_hard
        if loss_hard.numel() < n_min:
            loss_hard, _ = loss.topk(n_min)
        return torch.mean(loss_hard)


def weighted_dice_loss(
        prediction,
        target_seg,
        gt_num,
        index_mask,
        instance_num: int = 0,
        weighted_val: float = 1.0,
        weighted_num: int = 1,
        mode: str = "thing",
        reduction: str = "sum",
        eps: float = 1e-8,
):
    """
    Weighted version of Dice Loss used in PanopticFCN for multi-positive optimization.

    Args:
        prediction: prediction for Things or Stuff,
        target_seg: segmentation target for Things or Stuff,
        gt_num: ground truth number for Things or Stuff,
        index_mask: positive index mask for Things or Stuff,
        instance_num: instance number of Things or Stuff,
        weighted_val: values of k positives,
        weighted_num: number k for weighted loss,
        mode: used for things or stuff,
        reduction: 'none' | 'mean' | 'sum'
                   'none': No reduction will be applied to the output.
                   'mean': The output will be averaged.
                   'sum' : The output will be summed.
        eps: the minimum eps,
    """


    # avoid Nan. the other loss is need.
    if gt_num == 0:
        loss = prediction[0][0].sigmoid().mean() + eps
        return loss * gt_num

    n, _, h, w = target_seg.shape
    if mode == "thing":
        prediction = prediction.reshape(n, instance_num, weighted_num, h, w)
        prediction = prediction.reshape(-1, weighted_num, h, w)[index_mask, ...]  #instance_num, weight_num, h, w
        target_seg = target_seg.unsqueeze(2).expand(n, instance_num, weighted_num, h, w)  #
        target_seg = target_seg.reshape(-1, weighted_num, h, w)[index_mask, ...]
        weighted_val = weighted_val.reshape(-1, weighted_num)[index_mask, ...]
        weighted_val = weighted_val / torch.clamp(weighted_val.sum(dim=-1, keepdim=True), min=eps)
        prediction = torch.sigmoid(prediction)
        prediction = prediction.reshape(int(gt_num), weighted_num, h * w)
        target_seg = target_seg.reshape(int(gt_num), weighted_num, h * w)

    elif mode == "stuff":
        prediction = prediction.reshape(-1, h, w)[index_mask, ...]
        target_seg = target_seg.reshape(-1, h, w)[index_mask, ...]
        prediction = torch.sigmoid(prediction)
        prediction = prediction.reshape(int(gt_num), h * w)
        target_seg = target_seg.reshape(int(gt_num), h * w)
    else:
        raise ValueError

    # calculate dice loss
    loss_part = (prediction ** 2).sum(dim=-1) + (target_seg ** 2).sum(dim=-1)
    loss = 1 - 2 * (target_seg * prediction).sum(dim=-1) / torch.clamp(loss_part, min=eps)
    # normalize the loss
    loss = loss * weighted_val

    if reduction == "sum":
        loss = loss.sum()
    elif reduction == "mean":
        loss = loss.mean()
    return loss


def w_dice_loss(prediction, target_seg, weighted_val=None, reduction: str = "sum", eps: float = 1e-8):
    # calculate dice loss
    loss_part = (prediction ** 2).sum(dim=-1) + (target_seg ** 2).sum(dim=-1)
    loss = 1 - 2 * (target_seg * prediction).sum(dim=-1) / torch.clamp(loss_part, min=eps)
    # normalize the loss

    if weighted_val is not None:
        loss = loss * weighted_val

    if reduction == "sum":
        loss = loss.sum()
    elif reduction == "mean":
        loss = loss.mean()
    return loss



@SPARSE_INST_CRITERION_REGISTRY.register()
class SparseInstCriterion(nn.Module):
    # This part is partially derivated from: https://github.com/facebookresearch/detr/blob/main/models/detr.py

    def __init__(self, cfg, matcher):
        super().__init__()
        self.matcher = matcher
        self.losses = cfg.MODEL.SPARSE_INST.LOSS.ITEMS
        self.weight_dict = self.get_weight_dict(cfg)
        self.num_classes = cfg.MODEL.SPARSE_INST.DECODER.NUM_CLASSES

    def get_weight_dict(self, cfg):
        losses = ("loss_ce", "loss_mask", "loss_dice", "loss_objectness")
        weight_dict = {}
        ce_weight = cfg.MODEL.SPARSE_INST.LOSS.CLASS_WEIGHT
        mask_weight = cfg.MODEL.SPARSE_INST.LOSS.MASK_PIXEL_WEIGHT
        dice_weight = cfg.MODEL.SPARSE_INST.LOSS.MASK_DICE_WEIGHT
        objectness_weight = cfg.MODEL.SPARSE_INST.LOSS.OBJECTNESS_WEIGHT

        weight_dict = dict(
            zip(losses, (ce_weight, mask_weight, dice_weight, objectness_weight)))
        return weight_dict

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def loss_labels(self, outputs, targets, indices, num_instances, input_shape=None):
        assert "pred_logits" in outputs
        src_logits = outputs['pred_logits']
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        src_logits = src_logits.flatten(0, 1)
        # prepare one_hot target.
        target_classes = target_classes.flatten(0, 1)
        pos_inds = torch.nonzero(target_classes != self.num_classes, as_tuple=True)[0]
        labels = torch.zeros_like(src_logits)
        labels[pos_inds, target_classes[pos_inds]] = 1
        # comp focal loss.
        class_loss = sigmoid_focal_loss_jit(
            src_logits,
            labels,
            alpha=0.25,
            gamma=2.0,
            reduction="sum",
        ) / num_instances
        losses = {'loss_ce': class_loss}
        return losses
    
    def loss_masks_with_iou_objectness(self, outputs, targets, indices, num_instances, input_shape):
        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        # Bx100xHxW
        assert "pred_masks" in outputs
        assert "pred_scores" in outputs
        src_iou_scores = outputs["pred_scores"]
        src_masks = outputs["pred_masks"]
        with torch.no_grad():
            target_masks, _ = nested_masks_from_list(
                [t["masks"].tensor for t in targets], input_shape).decompose()
        num_masks = [len(t["masks"]) for t in targets]
        target_masks = target_masks.to(src_masks)
        if len(target_masks) == 0:
            losses = {
                "loss_dice": src_masks.sum() * 0.0,
                "loss_mask": src_masks.sum() * 0.0,
                "loss_objectness": src_iou_scores.sum() * 0.0
            }
            return losses

        src_masks = src_masks[src_idx]
        target_masks = F.interpolate(
            target_masks[:, None], size=src_masks.shape[-2:], mode='bilinear', align_corners=False).squeeze(1)

        src_masks = src_masks.flatten(1)
        # FIXME: tgt_idx
        mix_tgt_idx = torch.zeros_like(tgt_idx[1])
        cum_sum = 0
        for num_mask in num_masks:
            mix_tgt_idx[cum_sum: cum_sum + num_mask] = cum_sum
            cum_sum += num_mask
        mix_tgt_idx += tgt_idx[1]

        target_masks = target_masks[mix_tgt_idx].flatten(1)

        with torch.no_grad():
            ious = compute_mask_iou(src_masks, target_masks)


        tgt_iou_scores = ious
        src_iou_scores = src_iou_scores[src_idx]
        tgt_iou_scores = tgt_iou_scores.flatten(0)
        src_iou_scores = src_iou_scores.flatten(0)

        losses = {
            "loss_objectness": F.binary_cross_entropy_with_logits(src_iou_scores, tgt_iou_scores, reduction='mean'),
            "loss_dice": dice_loss(src_masks, target_masks) / num_instances,
            "loss_mask": F.binary_cross_entropy_with_logits(src_masks, target_masks, reduction='mean')
        }
        return losses


    def get_loss(self, loss, outputs, targets, indices, num_instances, **kwargs):
        loss_map = {
            "labels": self.loss_labels,
            "masks": self.loss_masks_with_iou_objectness,
        }
        if loss == "loss_objectness":
            # NOTE: loss_objectness will be calculated in `loss_masks_with_iou_objectness`
            return {}
        assert loss in loss_map
        return loss_map[loss](outputs, targets, indices, num_instances, **kwargs)

    def forward(self, outputs, targets, input_shape):

        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets, input_shape)
        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_instances = sum(len(t["labels"]) for t in targets)
        num_instances = torch.as_tensor(
            [num_instances], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_instances)
        num_instances = torch.clamp(num_instances / get_world_size(), min=1).item()
        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices,
                                        num_instances, input_shape=input_shape))

        for k in losses.keys():
            if k in self.weight_dict:
                losses[k] *= self.weight_dict[k]

        return losses



@SPARSE_INST_MATCHER_REGISTRY.register()
class SparseInstMatcherV1(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.alpha = cfg.MODEL.SPARSE_INST.MATCHER.ALPHA
        self.beta = cfg.MODEL.SPARSE_INST.MATCHER.BETA
        self.mask_score = dice_score

    @torch.no_grad()
    def forward(self, outputs, targets, input_shape):
        B, N, H, W = outputs["pred_masks"].shape
        pred_masks = outputs['pred_masks']
        pred_logits = outputs['pred_logits'].sigmoid()

        indices = []

        for i in range(B):
            tgt_ids = targets[i]["labels"]
            # no annotations
            if tgt_ids.shape[0] == 0:
                indices.append((torch.as_tensor([]),
                                torch.as_tensor([])))
                continue

            tgt_masks = targets[i]['masks'].tensor.to(pred_masks)
            pred_logit = pred_logits[i]
            out_masks = pred_masks[i]

            # upsampling:
            # (1) padding/
            # (2) upsampling to 1x input size (input_shape)
            # (3) downsampling to 0.25x input size (output mask size)
            ori_h, ori_w = tgt_masks.size(1), tgt_masks.size(2)
            tgt_masks_ = torch.zeros(
                (1, tgt_masks.size(0), input_shape[0], input_shape[1])).to(pred_masks)
            tgt_masks_[0, :, :ori_h, :ori_w] = tgt_masks
            tgt_masks = F.interpolate(
                tgt_masks_, size=out_masks.shape[-2:], mode='bilinear', align_corners=False)[0]

            # compute dice score and classification score
            tgt_masks = tgt_masks.flatten(1)
            out_masks = out_masks.flatten(1)

            mask_score = self.mask_score(out_masks, tgt_masks)
            # Nx(Number of gts)
            matching_prob = pred_logit[:, tgt_ids]
            C = (mask_score ** self.alpha) * (matching_prob ** self.beta)
            # hungarian matching
            inds = linear_sum_assignment(C.cpu(), maximize=True)
            indices.append(inds)
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


@SPARSE_INST_MATCHER_REGISTRY.register()
class SparseInstMatcher(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.alpha = cfg.MODEL.SPARSE_INST.MATCHER.ALPHA
        self.beta = cfg.MODEL.SPARSE_INST.MATCHER.BETA
        self.mask_score = dice_score

    def forward(self, outputs, targets, input_shape):
        with torch.no_grad():
            B, N, H, W = outputs["pred_masks"].shape
            pred_masks = outputs['pred_masks']
            pred_logits = outputs['pred_logits'].sigmoid()

            tgt_ids = torch.cat([v["labels"] for v in targets])

            if tgt_ids.shape[0] == 0:
                return [(torch.as_tensor([]).to(pred_logits), torch.as_tensor([]).to(pred_logits))] * B
            tgt_masks, _ = nested_masks_from_list(
                [t["masks"].tensor for t in targets], input_shape).decompose()
            device = pred_masks.device
            tgt_masks = tgt_masks.to(pred_masks)

            tgt_masks = F.interpolate(
                tgt_masks[:, None], size=pred_masks.shape[-2:], mode="bilinear", align_corners=False).squeeze(1)

            pred_masks = pred_masks.view(B * N, -1)
            tgt_masks = tgt_masks.flatten(1)

            mask_score = self.mask_score(pred_masks, tgt_masks)
            # Nx(Number of gts)
            matching_prob = pred_logits.view(B * N, -1)[:, tgt_ids]
            C = (mask_score ** self.alpha) * (matching_prob ** self.beta)
            C = C.view(B, N, -1).cpu()
            # hungarian matching
            sizes = [len(v["masks"]) for v in targets]
            indices = [linear_sum_assignment(c[i], maximize=True)
                       for i, c in enumerate(C.split(sizes, -1))]
            indices = [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(
                j, dtype=torch.int64)) for i, j in indices]
            return indices



def build_sparse_inst_matcher(cfg):
    name = cfg.MODEL.SPARSE_INST.MATCHER.NAME
    return SPARSE_INST_MATCHER_REGISTRY.get(name)(cfg)


def build_sparse_inst_criterion(cfg):
    matcher = build_sparse_inst_matcher(cfg)
    name = cfg.MODEL.SPARSE_INST.LOSS.NAME
    return SPARSE_INST_CRITERION_REGISTRY.get(name)(cfg, matcher)
