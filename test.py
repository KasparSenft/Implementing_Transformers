import torch
from modelling.attention import SelfAttentionHead

# n=26

# mdl = SelfAttentionHead(n)

# x = torch.rand(23,n)

# print(mdl(x).shape)

QUERY = torch.tensor([
    [[1.9269, 1.4873, 0.9007, -2.1055],
     [0.6784, -1.2345, -0.0431, -1.6047],
     [0.3559, -0.6866, -0.4934, 0.2415]],
    [[-1.1109, 0.0915, -2.3169, -0.2168],
     [-0.3097, -0.3957, 0.8034, -0.6216],
     [0.0000, 0.0000, 0.0000, 0.0000]]
])

QUERY_ATTENTION_MASK = torch.tensor([[1, 1, 1], [1, 1, 0]])



print(QUERY_ATTENTION_MASK.shape)