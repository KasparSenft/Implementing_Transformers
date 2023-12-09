from modelling.attention import MultiHeadAttention
import torch

x = torch.rand(2,10,28)
attn_mask = torch.ones(2,10)

mha = MultiHeadAttention(d_model=28, h =15)

print(mha(x,x,x, attn_mask).shape)