import numpy as np
import torch
from torch import nn
import math

offset_ranges = [(-18, 0), (-18, 0)]
n_samples = 9

offsets = [[np.random.randint(orange[0], orange[1]) for orange in offset_ranges]
           for _ in range(n_samples)]


print(offsets)

mask = torch.randn(4,4,4)

indices = torch.nonzero(mask, as_tuple=False)

ind = np.random.randint(len(indices[0]))

print(ind)


class Gaussian(nn.Module):
    def __init__(self, delta_var=0.3, pmaps_threshold=0.9):
        super().__init__()
        self.delta_var = delta_var
        # dist_var^2 = -2*sigma*ln(pmaps_threshold)
        self.two_sigma = delta_var * delta_var / (-math.log(pmaps_threshold))

    def forward(self, dist_map):
        return torch.exp(- dist_map * dist_map / self.two_sigma)

embeddings = torch.randn(4, 16)

anchor_emb = torch.randn(1, 16)

distance_map = torch.norm(embeddings - anchor_emb, dim=-1)
print("distance_map", distance_map)


a = -0.04/math.log(0.9)

a = -0.25/math.log(0.9)

print(a)

d = math.exp(-4/2.37)

print(d)

b = math.exp(-1/0.38)

#print(b)




#print(indices)


n = [0,0,0]

for i in range(2):
    for j in range(2):
        n[j] = n[i] +1



print("n:",n )