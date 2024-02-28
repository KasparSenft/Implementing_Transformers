import torch
from torch import nn
import torch.nn.functional as F
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
    def __init__(self,vocab_size, d_model, num_heads, num_layers,dim_feed_forward, dropout,max_len):
        super().__init__()

        #backbone
        self.Encoder = TransformerEncoder(vocab_size, d_model, num_heads, num_layers,dim_feed_forward, dropout,max_len)
        self.Decoder = TransformerDecoder(vocab_size, d_model, num_heads, num_layers,dim_feed_forward, dropout,max_len)

        #share weights
        self.embedding = nn.Embedding(vocab_size,d_model)
        self.Encoder.embedding = self.embedding
        self.Decoder.embedding = self.embedding


        #head
        self.linear = nn.Linear(d_model, vocab_size)
        self.linear.weight = self.embedding.weight

        #Save input
        self.max_len = max_len

        

    def forward(self, x_enc, enc_attn_mask, x_dec, dec_attn_mask):

        enc_out = self.Encoder(x_enc, enc_attn_mask)
        dec_out = self.Decoder(x_dec, dec_attn_mask, enc_out, enc_attn_mask)
        dec_out = self.linear(dec_out)

        return dec_out



from loguru import logger

class AutoRegressiveGenerator(object):
    def __init__(self, model, max_tokens, src_tokenizer,trgt_tokenizer, sampling = 'greedy'):
        """
        model: The trained transformer model to be used for generation
        max_tokens: maximum number of tokens to be generated in response
        eos_token: The eos token in the vocabulary
        src_tokenizer: the tokenizer used for the source language
        trgt_tokenizer: the tokenizer used for the target language
        """

        super().__init__()
        self.model = model
        self.max_len = self.model.max_len
        self.max_tokens = max_tokens
        self.eos = trgt_tokenizer.eos_token_id
        self.bos = trgt_tokenizer.bos_token_id
        self.src_tokenizer = src_tokenizer
        self.trgt_tokenizer = trgt_tokenizer
        self.sampling = sampling


    def tokenize_input(self, input):
        "Take an input in natural language and tokenize into format for model"
        tokens = self.src_tokenizer(input, max_length = self.max_len, padding = 'max_length')
        src_tokens, src_mask = torch.tensor(list(tokens.values()))

        #Add a batch dimension
        src_tokens, src_mask = src_tokens.unsqueeze(dim=0),src_mask.unsqueeze(dim=0)

        return src_tokens,src_mask


    def generate(self, input, max_tokens = None):

        if max_tokens is None:
            max_tokens = self.max_tokens

        #Get input tokens for encoder and get encoder out
        src_tokens, src_mask = self.tokenize_input(input)
        encoder_out = self.model.Encoder(src_tokens, src_mask)
        
        output_tokens = torch.tensor([self.bos])

        for i in range(self.max_tokens):


            #get decoder attention mask
            trgt_mask = F.pad(torch.ones(len(output_tokens)), (0,self.max_len - len(output_tokens)), value=0)
            trgt_mask = trgt_mask.unsqueeze(dim=0)

            #pad output sequence
            trgt_tokens = F.pad(output_tokens, (0,self.max_len - len(output_tokens)), value = self.trgt_tokenizer.pad_token_id)
            trgt_tokens = trgt_tokens.unsqueeze(dim=0)



            assert trgt_tokens.shape == trgt_mask.shape, 'Decoder input and decoder mask not same shape'
            assert trgt_tokens.shape == src_tokens.shape, 'Encoder input and decoder input not same shape'

            #Forward pass through the decoder
            decoder_out = self.model.Decoder(trgt_tokens, trgt_mask, encoder_out,src_mask)
            decoder_out = self.model.linear(decoder_out).softmax(dim=-1)
            

            #get new token
            if self.sampling == 'greedy':
                preds = decoder_out.argmax(dim=-1)
                new_token = preds[0,i]
            elif self.sampling == 'random':
                top_10, top_10_ids = decoder_out[0,i].topk(k=10, dim = -1)
                top_10 = top_10/top_10.sum()
                new_token = top_10_ids[top_10.multinomial(1)]
            else:
                raise NotImplementedError
            
            


            #check if eos_token
            if new_token == self.trgt_tokenizer.eos_token_id:
                break
            
            #add to output_tokens 
            output_tokens = torch.cat([output_tokens, torch.tensor([new_token])])


        #Decode output and return 
        decoded_out = self.trgt_tokenizer.decode(output_tokens, skip_special_tokens = True)

        return decoded_out, output_tokens
            










