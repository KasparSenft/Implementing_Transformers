import torch

def Attention(q,k,v,pad_mask=None):

    d = q.shape[-1]
    
    #make mask for lookahead
    forward_mask = torch.triu(torch.ones(d,d), diagonal=1)
    forward_mask.masked_fill_(forward_mask ==1, -torch.inf)

    #get indices of pad tokens
    pad_idxs = (pad_mask ==0).nonzero()

    #calculate scaling value
    scale = torch.tensor(d).sqrt()

    #pre-softmax values
    att_logits = torch.matmul(q,k.T)/scale

    #masking
    att_logits += forward_mask
    att_logits[pad_idxs].fill_(-torch.infty)


    w = torch.softmax(att_logits, dim =1) 
    return torch.matmul(w,v)

class SelfAttentionHead(torch.nn.Module):
    def __init__(self,embed_dim):
        """
        embed_dim (int): dimension of input embedding
        hidden_dim (int): dimension of hidden vectors
        """
        super().__init__()

        self.embed_dim = embed_dim
        self.qkv_proj = torch.nn.Linear(embed_dim, embed_dim*3)

    def forward(self,x):
        Q,K,V = torch.chunk(self.qkv_proj(x),3, dim = 1)
        out = Attention(Q,K,V)
    
        return out


