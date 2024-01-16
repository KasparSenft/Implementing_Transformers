import torch
from torch import nn
from modelling.attention import MultiHeadAttention, FFN

class BaseTransformerLayer(nn.Module):

    def __init__(self, input_dim, num_heads, feature_dim, dropout):
        """
        args:
            input_dim: int, dimension of model
            num_heads: int, number of attention heads
            feature_dim: int, hidden dimension of feed forward network
            dropout: float, percentage for dropout
        """

        super().__init__()


        self.self_attention = MultiHeadAttention(input_dim, num_heads, mask_future=False)
        self.feature_transformation = FFN(input_dim, feature_dim)
        self.layer_norm_1 = nn.LayerNorm(input_dim)
        self.layer_norm_2 = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(p = dropout)

    def forward(self, x, attn_mask):

        #Self Attention and Layer Norm
        sa_out = self.self_attention(x,x,x, attn_mask)
        sa_out = self.layer_norm_1(self.dropout(sa_out) + x)

        #FFN and layer norm
        out = self.feature_transformation(sa_out)
        out = self.layer_norm_2(self.dropout(out) + sa_out)

        return out 

class TransformerDecoderLayer(nn.Module):

    def __init__(self,input_dim, num_heads,feature_dim, dropout):
        """
        args:
            input_dim: int, dimension of model
            num_heads: int, number of attention heads
            feature_dim: int, hidden dimension of feed forward network
            dropout: float, percentage for dropout
        """

        super().__init__()

        self.self_attention = MultiHeadAttention(input_dim,num_heads,mask_future=True)
        self.encoder_attention = MultiHeadAttention(input_dim, num_heads, mask_future=False)
        self.feature_transformation = FFN(input_dim, feature_dim)
        self.layer_norm_1 = nn.LayerNorm(input_dim)
        self.layer_norm_2 = nn.LayerNorm(input_dim)
        self.layer_norm_3 = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, encoder_out, encoder_attn_mask, attn_mask):
        
        #Self Attention and Residual + Norm
        sa_out = self.self_attention(x,x,x,attn_mask)
        sa_out = self.layer_norm_1(self.dropout(sa_out) + x)

        #Cross Attention and Residual + Norm
        ca_out = self.encoder_attention(sa_out, encoder_out, encoder_out, encoder_attn_mask)
        ca_out = self.layer_norm_2(self.dropout(ca_out) + sa_out)

        #FFN and Residual + Norm
        out = self.feature_transformation(ca_out)
        out = self.layer_norm_3(self.dropout(out) + ca_out)
        return out 
