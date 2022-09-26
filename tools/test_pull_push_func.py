import torch

a = torch.randn(10, 5, 1, 20)
b = torch.randn(10, 1, 7, 20)

similarity = torch.norm(a - b, dim=-1)

print(similarity.shape)

a = torch.tensor([1., 2, 3, 4])

a = a.unsqueeze(1).expand(4, 4)

print(a)


a = torch.tensor([1, 2, 3, 4, 5, 6])

dis = a.unsqueeze(1) - a.unsqueeze(0)

b = a.unsqueeze(1)

b = b.expand([b.shape[0], b.shape[0]])

#print(b)

#print(dis)


def pull_push_loss(cos_maps, gt_instance, index_mask):
    """
    cos_maps shape: b, n, h, w
    gt_instance shape: b, n, w, h
    example:
    [1,2,3,4]

    unsqueeze(1)
    pos ={
        1 1 1 1
        2 2 2 2
        3 3 3 3
        4 4 4 4
        }
    unsqueeze(0)
    neg ={
        1 2 3 4
        1 2 3 4
        1 2 3 4
        1 2 3 4
    }


    return:

    """


    b_size, inst_num, h ,w = cos_maps.shape

    neg_index = index_mask.unsqueeze(1).expand(b_size, inst_num, inst_num)

    pos_index = index_mask.unsqueeze(2).expand(b_size, inst_num, inst_num)

    mask = neg_index*pos_index

    mask = mask.triu(1)

    index = torch.nonzero(mask)

    neg_maps = cos_maps.unsqueeze(1).expand(b_size, inst_num, inst_num, h ,w)

    pos_maps = cos_maps.unsqueeze(2).expand(b_size, inst_num, inst_num, h ,w)

    dis = neg_maps - pos_maps + 0.5

    dis = dis.clamp(min=0.0)


    instance_masks = gt_instance.unsqueeze(2).expand(b_size, inst_num, inst_num, h ,w)

    instance_masks = instance_masks[index[:,0], index[:,1], index[:,2], :, :]

    loss = dis[index[:,0], index[:,1], index[:,2], :, :]*instance_masks

    loss = -torch.log(1-loss/2.5)

    loss = loss.sum()

    return loss




cos_maps = torch.zeros(2, 3, 4, 4)

gt_instance = torch.ones(2, 3, 4, 4)

index_mask = torch.zeros(2, 3)

index_mask[0, :] = 1


index_mask[1, 0:2] = 1


gt_instance[1, 2, :,:] = 0


loss = pull_push_loss(cos_maps, gt_instance, index_mask)


#print(loss)



