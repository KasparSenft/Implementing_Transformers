import torch
from torch import nn
from modelling.attention import MultiHeadAttention, FFN
from modelling.embeddings import PositionalEncoding


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

    def forward(self, x, attn_mask, encoder_out, encoder_attn_mask):
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


class TransformerEncoder(nn.Module):
    def __init__(self,vocab_size, d_model, num_heads, num_encoder_layers,dim_feed_forward, dropout,max_len):
        
        super().__init__()
    
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size,d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len=max_len)
        self.dropout = nn.Dropout(dropout)

        self.enc_blocks = nn.ModuleList([
            BaseTransformerLayer(d_model,num_heads,dim_feed_forward, dropout) for _ in range(num_encoder_layers)
        ])

    def forward(self, x, attn_mask):
            #embed input
            x = self.embedding(x) * torch.tensor(self.d_model).sqrt()
            x = self.pos_encoding(x)
            x = self.dropout(x)

            for block in self.enc_blocks:
                out = block(x, attn_mask)
            return out
    

class TransformerDecoder(nn.Module):
    def __init__(self,vocab_size, d_model, num_heads, num_decoder_layers,dim_feed_forward, dropout,max_len):
        super().__init__()

        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size,d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len=max_len)
        self.dropout = nn.Dropout(dropout)
        self.dec_blocks = nn.Sequential()


        self.enc_blocks = nn.ModuleList([
            TransformerDecoderLayer(d_model,num_heads, dim_feed_forward,dropout) for _ in range(num_decoder_layers)
        ])


    def forward(self, x, attn_mask, enc_outs, encoder_attn_mask):
        #embed input
        x_embed = self.embedding(x) * torch.tensor(self.d_model).sqrt()
        x_pos = self.pos_encoding(x_embed)
        x = self.dropout(x_pos)

        for block in self.dec_blocks:
            x = block(x, attn_mask, enc_outs,encoder_attn_mask)

        return x


class Transformer(nn.Module):
    def __init__(self,vocab_size, d_model, num_heads, num_decoder_layers,dim_feed_forward, dropout,max_len):
        super().__init__()

        #backbone
        self.Encoder = TransformerEncoder(vocab_size, d_model, num_heads, num_decoder_layers,dim_feed_forward, dropout,max_len)
        self.Decoder = TransformerDecoder(vocab_size, d_model, num_heads, num_decoder_layers,dim_feed_forward, dropout,max_len)

        #share weights
        self.embedding = nn.Embedding(vocab_size,d_model)
        self.Encoder.embedding = self.embedding
        self.Decoder.embedding = self.embedding


        #head
        self.linear = nn.Linear(d_model, vocab_size)
        self.linear.weight = self.embedding.weight

        

    def forward(self, x_enc, enc_attn_mask, x_dec, dec_attn_mask):

        enc_out = self.Encoder(x_enc, enc_attn_mask)
        dec_out = self.Decoder(x_dec, dec_attn_mask, enc_out, enc_attn_mask)
        dec_out = self.linear(dec_out)

        return dec_out
