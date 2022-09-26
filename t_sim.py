import torch

def get_low_simi_feats(feats, feats_mask, feats_num, topk_num=1):
    """
    b, inst_num, 100, dim
    b, inst_num, 100
    b, inst_num, 100
    """
    feats_mask[feats_mask==0] = 2
    feats_mask[feats_mask == 1] = 0


    a_n = feats.norm(dim=-1).unsqueeze(-1)

    a_norm = feats / a_n.clamp(min=1e-8)

    similarity = torch.matmul(a_norm, a_norm.transpose(-2, -1))

    similarity = similarity.triu(diagonal=1)

    cal_feat = torch.cumsum(similarity, dim=-2)

    cal_num = cal_feat.diagonal(dim1=-2, dim2=-1)  #b, inst_num, 100

    mean_num = cal_num/(feats_num+1e-8)

    mean_num = mean_num+feats_mask

    value, guided_index = mean_num.topk(topk_num, dim=-1, largest=False)

    return guided_index



feats = torch.randn(2, 2, 4, 64)

feats_mask = torch.randn(2, 2, 4)

feats_mask[feats_mask>0]=1
feats_mask[feats_mask<0.5]=0

feats_num = torch.arange(0, 4)

feats_num[0] = 1e-8

feats_num = feats_num.unsqueeze(0)
feats_num = feats_num.unsqueeze(0)
feats_num.expand(2, 2, 4)


index = get_low_simi_feats(feats, feats_mask, feats_num)

index = index.unsqueeze(-1)
index = index.expand(2,2,1,64)
print(index)

#get_low_simi_feat(feats, feats_mask)
idx_feat_th = torch.gather(feats, dim=2, index=index)

#print(idx_feat_th)

feats = torch.randn(1,2,3,64)

a_n = feats.norm(dim=-1).unsqueeze(-1)

a_norm = feats / a_n.clamp(min=1e-8)

similarity = torch.matmul(a_norm, a_norm.transpose(-2, -1))

print(similarity)

print(similarity.shape)

print(similarity.triu(diagonal=1))



"""
idx_feat_th = torch.gather(y1, dim=2, index=guided_index)
print(idx_feat_th)


guided_index = guided_index.reshape(batch_num, -1)  # b, thing_num*7

inst_w = predict_embedding_vector.reshape(batch_num, v_num, -1)  # batch_num, v_num, h*w

idx_inst = guided_index.unsqueeze(1).expand(batch_num, v_num, -1)  # b,1,thing_num*7 --- batch_num, v_num, thing_num*7
idx_feat_th = torch.gather(inst_w, dim=2, index=idx_inst)  # batch_num, v_num, thing_num*7
idx_feat_th = idx_feat_th.permute(0, 2, 1)  # batch_num,  thing_num*7, v_num
"""



