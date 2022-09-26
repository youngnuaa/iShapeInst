# Copyright (c) Tianheng Cheng and its affiliates. All Rights Reserved

import torch
import torch.nn as nn
import torch.nn.functional as F


from detectron2.structures import ImageList, Instances, BitMasks
from detectron2.modeling import META_ARCH_REGISTRY, build_backbone

from .encoder import build_sparse_inst_encoder
from .decoder import build_sparse_inst_decoder
from .loss import build_sparse_inst_criterion
from .utils import nested_tensor_from_tensor_list
from .gt_generate import GenerateGT
from .loss import sigmoid_focal_loss, w_dice_loss, l2_loss, \
    OhemCELoss, EuSoftMaskLoss, SoftMaskLoss, PullPushLoss, \
    L2Loss, PushLoss, PullLosee, pull_push_loss

__all__ = ["SparseInst"]


@torch.jit.script
def rescoring_mask(scores, mask_pred, masks):
    mask_pred_ = mask_pred.float()
    return scores * ((masks * mask_pred_).sum([1, 2]) / (mask_pred_.sum([1, 2]) + 1e-6))


@META_ARCH_REGISTRY.register()
class SparseInst(nn.Module):

    def __init__(self, cfg):
        super().__init__()

        # move to target device
        self.device = torch.device(cfg.MODEL.DEVICE)

        # backbone
        self.backbone = build_backbone(cfg)
        self.size_divisibility = self.backbone.size_divisibility
        output_shape = self.backbone.output_shape()

        # encoder & decoder
        self.encoder = build_sparse_inst_encoder(cfg, output_shape)
        self.decoder = build_sparse_inst_decoder(cfg)

        # matcher & loss (matcher is built in loss)
        #self.criterion = build_sparse_inst_criterion(cfg)

        # data and preprocessing
        self.mask_format = cfg.INPUT.MASK_FORMAT

        self.pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).to(self.device).view(3, 1, 1)
        self.pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).to(self.device).view(3, 1, 1)
        # self.normalizer = lambda x: (x - pixel_mean) / pixel_std

        # inference
        self.cls_threshold = cfg.MODEL.SPARSE_INST.CLS_THRESHOLD
        self.mask_threshold = cfg.MODEL.SPARSE_INST.MASK_THRESHOLD
        self.max_detections = cfg.MODEL.SPARSE_INST.MAX_DETECTIONS
        self.focal_loss_alpha      = cfg.MODEL.SPARSE_INST.FOCAL_LOSS_ALPHA
        self.focal_loss_gamma      = cfg.MODEL.SPARSE_INST.FOCAL_LOSS_GAMMA
        self.margin      = cfg.MODEL.SPARSE_INST.LOSS.PULL_PUSH_MARGIN
        self.soft_mask_t = cfg.MODEL.SPARSE_INST.LOSS.SOFT_MASK_T
        self.pos_num = cfg.MODEL.SPARSE_INST.LOSS.POS_NUM
        self.topk_num = cfg.MODEL.SPARSE_INST.LOSS.TOPK_NUM
        self.get_ground_truth = GenerateGT(cfg)
        self.soft_mask_loss = SoftMaskLoss(pos_num=self.pos_num)


    
    def normalizer(self, image):
        image = (image - self.pixel_mean) / self.pixel_std
        return image


    def preprocess_inputs(self, batched_inputs):
        images = [x["image"].to(self.device) for x in batched_inputs]

        images = [self.normalizer(x) for x in images]
        images = ImageList.from_tensors(images, 32)

        return images


    def prepare_targets(self, targets):
        new_targets = []
        for targets_per_image in targets:
            target = {}
            gt_classes = targets_per_image.gt_classes
            target["labels"] = gt_classes.to(self.device)
            h, w = targets_per_image.image_size
            if not targets_per_image.has('gt_masks'):
                gt_masks = BitMasks(torch.empty(0, h, w))
            else:
                gt_masks = targets_per_image.gt_masks
                if self.mask_format == "polygon":
                    if len(gt_masks.polygons) == 0:
                        gt_masks = BitMasks(torch.empty(0, h, w))
                    else:
                        gt_masks = BitMasks.from_polygon_masks(gt_masks.polygons, h, w)

            target["masks"] = gt_masks.to(self.device)
            new_targets.append(target)

        return new_targets



    def forward(self, batched_inputs):

        images = self.preprocess_inputs(batched_inputs)

        if isinstance(images, (list, torch.Tensor)):
            images = nested_tensor_from_tensor_list(images)

        max_shape = images.tensor.shape[2:]
        # forward
        #print("images shape", images.tensor.shape)
        features = self.backbone(images.tensor)

        l_feat = self.encoder(features)

        output = self.decoder(l_feat)

        #print("pred_embedding shape", output["pred_embedding"].shape)
        #print("pred_masks shape", output["pred_masks"].shape)
        #print("pred_logits shape", output["pred_logits"].shape)

        if self.training:
            #gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
            #targets = self.prepare_targets(gt_instances)
            targets = self.get_ground_truth.generate(batched_inputs, output["pred_logits"], output["encode_feat"])

            losses = self.losses(output, targets)

            return losses
        else:
            results = self.inference(output, batched_inputs, max_shape, images.image_sizes)
            processed_results = [{"instances": r} for r in results]
            return processed_results


    def forward_test(self, images):
        pass


    def gen_instance(self, pred_masks, idx_feat_inst):
        """
        pred_masks shape = b, embedding_feat_c, h, w

        idx_feat_inst shape = b, inst_num, embedding_feat_c
        """

        #b, inst_num, h, w
        return


    def cal_similarity(self, base_w, anchor_w, sim_type="cosine"):
        """
        base_w shape: b, n, dim
        anchor_w shape: b, dim, h ,w

        return: b, n, m
        """
        b, dim, h, w = anchor_w.shape
        _, n, _ = base_w.shape
        anchor_w = anchor_w.reshape(b, dim, -1)
        anchor_w = anchor_w.transpose(-2, -1) #b, h*w, dim

        if sim_type == "cosine":
            a_n, b_n = base_w.norm(dim=-1).unsqueeze(-1), anchor_w.norm(dim=-1).unsqueeze(-1)

            a_norm = base_w / a_n.clamp(min=1e-8)

            b_norm = anchor_w / b_n.clamp(min=1e-8)

            similarity = torch.matmul(a_norm, b_norm.transpose(-2, -1))
        elif sim_type == "L2":
            base_w = base_w.unsqueeze(2)      #b, n, 1, dim
            anchor_w = anchor_w.unsqueeze(1)  #b, 1, h*w, dim
            similarity = torch.norm(base_w - anchor_w, dim=-1) #b, n, h*w
        else:
            raise NotImplementedError

        similarity = similarity.reshape(b, n, h ,w)

        return similarity


    def get_pre_gt_map(self, pre_map, gt_instance, gt_index_mask):
        """
        pre_map shape: b, inst_num*pos_num, h, w
        gt_instance shape: b, inst_num, h, w
        gt_index_mask shape:b, inst_num
        """
        b, inst_pos_num, h, w = pre_map.shape
        _, inst_num, _, _ = gt_instance.shape
        #self.pos_num


        gt_instance = gt_instance.unsqueeze(2).expand(b,inst_num, self.pos_num, h, w)


    def losses(self, output, gt_dict):
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

        pred_mask = output["pred_logits"]
        embedding_feat = output["embedding_feat"]
        encode_feat = output["encode_feat"]

        gt_scoremaps = gt_dict["gt_scoremaps"]
        gt_instances = gt_dict["gt_instances"]
        gt_index_mask = gt_dict["gt_inst_nums"]
        gt_img_instances = gt_dict["gt_img_instances"]

        gt_range_nums = gt_dict["gt_range_nums"]

        """
        loss_pos_th = sigmoid_focal_loss(pred_mask, gt_scoremaps,
                                         mode="thing",
                                         alpha=self.focal_loss_alpha,
                                         gamma=self.focal_loss_gamma,
                                         reduction="sum")
        """

        """
        pred_mask = torch.sigmoid(pred_mask)

        loss_pos_th = self.ohem(pred_mask, gt_scoremaps)
        """

        index_mask = gt_index_mask.reshape(-1).bool()
        num = max(index_mask.reshape(-1).sum(), 1)

        loss_pos_th = sigmoid_focal_loss(pred_mask, gt_scoremaps,
                                         mode="stuff",
                                         alpha=self.focal_loss_alpha,
                                         gamma=self.focal_loss_gamma,
                                         reduction="sum")/num

        pred_mask = torch.sigmoid(pred_mask)


        thing_num = int(max(gt_index_mask.sum(dim=1).max(), 1))

        gt_index_mask = gt_index_mask[:, :thing_num]

        gt_instances = gt_instances[:, :thing_num]

        gt_img_instances = gt_img_instances[:, :thing_num]

        gt_range_nums = gt_range_nums[:, :thing_num]


        mean_embedding_vectors, _ = self.get_mean_embedding_vector(embedding_feat, gt_instances)


        idx_feat_th, weighted_values = self.get_simi_feat(pred_mask, embedding_feat, gt_instances, gt_range_nums)


        isntance_masks = self.get_isntance_mask(idx_feat_th, encode_feat, is_norm=True)  # batch_num, thing_num*7, H, W
        embedding_cossimi_masks = self.cal_similarity(idx_feat_th, embedding_feat)  #batch_num, thing_num*7, H, W
        mean_cossimi_masks = self.cal_similarity(mean_embedding_vectors, embedding_feat)  #batch_num, thing_num, H, W

        soft_loss = self.soft_mask_loss(embedding_cossimi_masks, gt_instances, gt_index_mask, weighted_val=weighted_values)

        p_loss = pull_push_loss(mean_cossimi_masks, gt_instances, gt_index_mask, ohem_num=50)
        prediction, target_seg, weighted_val = self.get_gt_pre_inst_mask(isntance_masks, gt_img_instances,
                                                                         gt_index_mask, self.pos_num, weighted_val=weighted_values)



        num = num*self.pos_num
        inst_loss = w_dice_loss(prediction, target_seg, weighted_val)/num



        loss = {}
        loss["loss_pos"] = 100*loss_pos_th
        #loss["embedding_l2_loss"] = pl_loss
        loss["p_loss"] = p_loss
        #loss["pl_loss"] = pl_loss
        #oss["ph_loss"] = ph_loss
        loss["inst_loss"] = 5000*inst_loss
        loss["soft_loss"] = 0.5*soft_loss
        #loss["mean_inst_loss"] = mean_inst_loss

        return loss


    def get_mean_embedding_vector(self, pred_weights, mask_maps):
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

    def cal_high_score_instance_mask(self, predict_mask, predict_embedding_vector, gt_embeddingmaps):
        """


        """
        batch_num, c_num, h, w = predict_mask.shape

        _, v_num, _, _ = predict_embedding_vector.shape

        #predict_mask = torch.sigmoid(predict_mask)

        pre_regions = predict_mask * gt_embeddingmaps  # batch_num, inst_num, h, w

        weighted_values, guided_index = torch.topk(pre_regions.reshape(*pre_regions.shape[:2], -1),
                                                   k=self.pos_num, dim=-1)  # b, 100, 7

        guided_index = guided_index.reshape(batch_num, -1)  # b, thing_num*7

        inst_w = predict_embedding_vector.reshape(batch_num, v_num, -1)  # batch_num, v_num, h*w

        idx_inst = guided_index.unsqueeze(1).expand(batch_num, v_num,-1)  # b,1,thing_num*7 --- batch_num, v_num, thing_num*7
        idx_feat_th = torch.gather(inst_w, dim=2, index=idx_inst)  # batch_num, v_num, thing_num*7
        idx_feat_th = idx_feat_th.permute(0, 2, 1)  # batch_num,  thing_num*7, v_num

        return idx_feat_th, weighted_values

    def get_isntance_mask(self, embedding_vectors, instance_vector_map, is_norm=False):
        """
        embedding_vectors: b,n,embedding_v
        instance_vector_map:b,embedding_v,h, w
        (b,n,embedding_v)*(b,embedding_v,h*w)
        """
        b, c, h, w = instance_vector_map.shape
        #print("instance_vector_map: ", instance_vector_map.shape)
        instance_vector_map = instance_vector_map.reshape(b, c, -1)

        if is_norm:
            a_n= embedding_vectors.norm(dim=-1).unsqueeze(-1)
            embedding_vectors = embedding_vectors / a_n.clamp(min=1e-8)

        isntance_mask = torch.matmul(embedding_vectors, instance_vector_map)
        #print("isntance_mask", isntance_mask.shape)
        isntance_mask = isntance_mask.reshape(b, -1, h, w)
        return isntance_mask

    def get_gt_pre_inst_mask(self, isntance_masks, gt_instances, gt_index_mask, weighted_num, weighted_val=None, eps=1e-8):
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

        if weighted_val is not None:
            weighted_val = weighted_val.reshape(-1, weighted_num)[index_mask, ...]
            #weighted_val = weighted_val / torch.clamp(weighted_val.sum(dim=-1, keepdim=True), min=eps)

        prediction = torch.sigmoid(prediction)
        prediction = prediction.reshape(int(gt_num), weighted_num, mask_h * mask_w)
        target_seg = target_seg.reshape(int(gt_num), weighted_num, mask_h * mask_w)

        return prediction, target_seg, weighted_val

    def get_gt_mean_inst_mask(self, mean_isntance_masks, gt_instances, gt_index_mask):
        """
        isntance_masks: batch_num, thing_num, H, W
        gt_instances:   batch_num, thing_num, H, W
        gt_inst_nums:   batch_num, thing_num
        """
        assert mean_isntance_masks.shape == gt_instances.shape

        index_mask = gt_index_mask.reshape(-1).bool()

        batch_num, inst_num, mask_h, mask_w = mean_isntance_masks.shape
        _, inst_num, _, _ = gt_instances.shape

        prediction = mean_isntance_masks.reshape(-1, mask_h, mask_w)[index_mask, ...]
        target_seg = gt_instances.reshape(-1, mask_h, mask_w)[index_mask, ...]

        prediction = torch.sigmoid(prediction)
        prediction = prediction.reshape(-1, mask_h * mask_w)
        target_seg = target_seg.reshape(-1, mask_h * mask_w)

        return prediction, target_seg

    def get_simi_feat(self, predict_mask, predict_embedding_vector, gt_embeddingmaps, gt_range_nums):
        """
        feats shape, b, inst_num, topk_num, dim, topk_num is score.
        """

        batch_num, c_num, h, w = predict_mask.shape

        _, d_num, _, _ = predict_embedding_vector.shape

        _, inst_num, _, _ = gt_embeddingmaps.shape

        # predict_mask = torch.sigmoid(predict_mask)

        pre_regions = predict_mask * gt_embeddingmaps  # batch_num, inst_num, h, w

        # b, inst_num, self.topk_num
        weighted_values, guided_index = torch.topk(pre_regions.reshape(*pre_regions.shape[:2], -1),
                                                   k=self.topk_num, dim=-1) #b, inst_num, 100

        guided_index = guided_index.reshape(batch_num, -1)  # b, inst_num*100

        inst_w = predict_embedding_vector.reshape(batch_num, d_num, -1)  # batch_num, d_num, h*w
        idx_inst = guided_index.unsqueeze(1).expand(batch_num, d_num,-1)  # b,1,inst_num*100 --- batch_num, dim_num, inst_num*100
        idx_feat_th = torch.gather(inst_w, dim=2, index=idx_inst)  # batch_num, dim_num, inst_num*100
        idx_feat_th = idx_feat_th.permute(0, 2, 1)  # batch_num,  inst_num*100, dim_num
        idx_feat_th = idx_feat_th.reshape(batch_num, inst_num, self.topk_num, d_num) #batch_num, inst_num, 100, dim_num

        inst_m = gt_embeddingmaps.reshape(*gt_embeddingmaps.shape[:2], -1)  # batch_num, inst_num, -1
        guided_index = guided_index.reshape(batch_num, inst_num, -1)
        idx_feat_mask = torch.gather(inst_m, dim=2, index=guided_index)  # batch_num, insnt_num, self.topk_num

        #idx_feat_mask = idx_feat_mask.reshape(batch_num, inst_num, self.topk_num) #batch_num, insnt_num, self.topk_num

        #print(idx_feat_mask)
        end_guided_index = self.get_low_simi_feats(idx_feat_th, idx_feat_mask, gt_range_nums) #b, inst_num, 7

        end_weighted_values = torch.gather(idx_feat_mask, dim=2, index=end_guided_index)  #b, inst_num, 7

        end_idx_inst = end_guided_index.unsqueeze(-1).expand(batch_num, inst_num, self.pos_num, d_num)

        #end_idx_inst = end_idx_inst.reshape(batch_num, -1, d_num) #b, inst_num*self.pos_num, d_num

        idx_feat_end_th = torch.gather(idx_feat_th, dim=2, index=end_idx_inst) ##b, inst_num, self.pos_num, d_num

        idx_feat_end_th = idx_feat_end_th.reshape(batch_num, -1, d_num)

        return idx_feat_end_th, end_weighted_values

    @torch.no_grad()
    def get_low_simi_feats(self, feats, feats_mask, feats_num):
        """
        b, inst_num, 100, dim
        b, inst_num, 100
        b, inst_num, 100
        """
        feats_mask = (1 - feats_mask) * 2
        a_n = feats.norm(dim=-1).unsqueeze(-1)

        a_norm = feats / a_n.clamp(min=1e-8)

        similarity = torch.matmul(a_norm, a_norm.transpose(-2, -1))

        similarity = similarity.triu(diagonal=1)

        cal_feat = torch.cumsum(similarity, dim=-2)

        cal_num = cal_feat.diagonal(dim1=-2, dim2=-1)  # b, inst_num, 100

        mean_num = cal_num / (feats_num + 1e-8)

        mean_num = mean_num + feats_mask

        #b, inst_num, 7
        value, guided_index = mean_num.topk(self.pos_num, dim=-1, largest=False)

        return guided_index  #


    def inference(self, output, batched_inputs, max_shape, image_sizes):
        # max_detections = self.max_detections
        results = []
        pred_scores = output["pred_logits"].sigmoid()
        pred_masks = output["pred_masks"].sigmoid()
        pred_objectness = output["pred_scores"].sigmoid()
        pred_scores = torch.sqrt(pred_scores * pred_objectness)
    
        for _, (scores_per_image, mask_pred_per_image, batched_input, img_shape) in enumerate(zip(
                pred_scores, pred_masks, batched_inputs, image_sizes)):

            ori_shape = (batched_input["height"], batched_input["width"])
            result = Instances(ori_shape)
            # max/argmax
            scores, labels = scores_per_image.max(dim=-1)
            # cls threshold
            keep = scores > self.cls_threshold
            scores = scores[keep]
            labels = labels[keep]
            mask_pred_per_image = mask_pred_per_image[keep]

            if scores.size(0) == 0:
                result.scores = scores
                result.pred_classes = labels
                results.append(result)
                continue

            h, w = img_shape
            # rescoring mask using maskness
            scores = rescoring_mask(scores, mask_pred_per_image > self.mask_threshold, mask_pred_per_image)

            # upsample the masks to the original resolution:
            # (1) upsampling the masks to the padded inputs, remove the padding area
            # (2) upsampling/downsampling the masks to the original sizes
            mask_pred_per_image = F.interpolate(
                mask_pred_per_image.unsqueeze(1), size=max_shape, mode="bilinear", align_corners=False)[:, :, :h, :w]
            mask_pred_per_image = F.interpolate(
                mask_pred_per_image, size=ori_shape, mode='bilinear', align_corners=False).squeeze(1)

            mask_pred = mask_pred_per_image > self.mask_threshold
            mask_pred = BitMasks(mask_pred)

            # using Detectron2 Instances to store the final results
            result.pred_masks = mask_pred
            result.scores = scores
            result.pred_classes = labels
            results.append(result)

        return results



    

