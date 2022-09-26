import torch


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



pre_weight = torch.randn(1,2,3,3)

mask_map = torch.zeros(1, 2, 3, 3)

print(pre_weight.reshape(1,2,-1))

mask_map[:,0, 0:3, 0:1] = 1

mask_map[:,1, 1:3, 1:2] = 1

#print(mask_map)
print(mask_map.reshape(1,2,-1))



mean_embedding_vectors, gt_inst_piexl_nums = get_mean_embedding_vector(pre_weight, mask_map)


print(mean_embedding_vectors)

print(mean_embedding_vectors.shape)

print(gt_inst_piexl_nums)