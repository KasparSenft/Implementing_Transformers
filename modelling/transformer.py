import torch
from attention import MultiHeadSelfAttention, FFN

class Encoder(torch.nn.module):
    def __init__(self, d_model, h=8, dff = 2048, p = 0.5):
        super().__init__()

        self.mha = MultiHeadSelfAttention(d_model, h)
        self.ffn = FFN(d_model, dff)
        self.ln1 = torch.nn.LayerNorm()
        self.ln2 = torch.nn.LayerNorm()
        self.drop = torch.nn.Dropout(p)

    def forward(self,x,attn_mask):

        out = self.mha(x,attn_mask) + x
        out = self.drop(self.ln1(out))

        out = self.ffn(x) + x
        out = self.drop(self.ln2(out))

        return out
    