import torch

class Attention(torch.nn.Module):
    def __init__(self, mask_future = False):
        super().__init__()
        self.mask_future = mask_future

    def forward(self, q,k,v,attn_mask):
        d = q.shape[-1] #embed dimension
        n_q = q.shape[1] #len of input
        n_k = k.shape[1] #len of output

        #make mask for lookahead
        forward_mask = torch.tril(torch.ones(n_q,n_k))
   
        #calculate scaling value
        scale = torch.tensor(d).sqrt()

        #pre-softmax values
        attn_logits = torch.bmm(q,k.transpose(1,2))/scale

        #masking
        if self.mask_future:
            attn_logits.masked_fill_(forward_mask == 0, -torch.inf)
        
        attn_mask = attn_mask.unsqueeze(dim=1)
        attn_logits.masked_fill_(attn_mask == 0, -torch.inf)

        w = torch.softmax(attn_logits, dim =-1) 
    
        return torch.bmm(w,v)
    
class AttentionHead(torch.nn.Module):
    def __init__(self,d_model, mask_future = False):
        """
        d_model (int): dimension of input embedding
        hidden_dim (int): dimension of hidden vectors
        mask_future (bool): whether to have a lookahead mask
        """
        super().__init__()

        self.d_model = d_model
        self.mask_future = mask_future
        self.attn = Attention(mask_future = mask_future)
        self.q = torch.nn.Linear(d_model, d_model, bias = False)
        self.k = torch.nn.Linear(d_model, d_model, bias = False)
        self.v = torch.nn.Linear(d_model, d_model, bias = False)

    def forward(self,x_q,x_k,x_v,attn_mask):

        Q = self.q(x_q)
        K = self.k(x_k)
        V = self.v(x_v)

        out = self.attn(Q,K,V, attn_mask)
    
        return out

class SelfAttentionHead(torch.nn.Module):
    def __init__(self,d_model, mask_future = False):
        """
        d_model (int): dimension of input embedding
        hidden_dim (int): dimension of hidden vectors
        mask_future (bool): whether to have a lookahead mask
        """
        super().__init__()
        self.AttetionHead = AttentionHead(d_model, mask_future)

    def forward(self,x,attn_mask):
        out = self.AttetionHead(x,x,x,attn_mask)
        return out

class FFN(torch.nn.Module):
    def __init__(self, d_model, dff):
        """
        d_model: embedding dimension of model
        dff: hidden dimension of feed forward netowrk
        """
        super().__init__()

        linear1 = torch.nn.Linear(d_model, dff)
        linear2 = torch.nn.Linear(dff, d_model)
        relu = torch.nn.ReLU()
        layers = [linear1,relu,linear2]

        self.ffn = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.ffn(x)
    

class MultiHeadAttention(torch.nn.Module):
    def __init__(self, d_model, h, mask_future = False):
        super().__init__()

        self.attn_heads = [AttentionHead(d_model,mask_future) for _ in range(h)]
        self.proj = torch.nn.Linear(h*d_model, d_model, bias = False)


    def forward(self,x_q,x_k,x_v, attn_mask):

        attns = []
        for head in self.attn_heads:
            attns.append(head(x_q,x_k,x_v,attn_mask))

        out = self.proj(torch.cat(attns, dim = -1))

        return out

class MultiHeadSelfAttention(torch.nn.Module):
    def __init__(self, d_model, h, mask_future = False):
        super().__init__()
        self.MHA = MultiHeadAttention(d_model,h,mask_future)

    def forward(self, x, attn_mask):
        out = self.MHA(x,x,x,attn_mask)
        return out