import torch
import numpy as np
import matplotlib.pyplot as plt



a = torch.tensor([0,0,1,1,1,2,2,4])

b = a.unsqueeze(1)
c = a.unsqueeze(0)

d = b-c

print(d)
d[d!=0] = 1
d = 2-d
print(d)
f = torch.triu(d, diagonal=1)
print(f)
g = f[f>0]
g = g-1
#g[g==1] = 0
#g[g==2] = 1
print(g)


#a =