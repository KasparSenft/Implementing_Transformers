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

class SelfAttentionHead(torch.nn.Module):
    def __init__(self,embed_dim, mask_future = False):
        """
        embed_dim (int): dimension of input embedding
        hidden_dim (int): dimension of hidden vectors
        mask_future (bool): whether to have a lookahead mask
        """
        super().__init__()

        self.embed_dim = embed_dim
        self.mask_future = mask_future
        self.attn = Attention(mask_future = mask_future)
        self.q = torch.nn.Linear(embed_dim, embed_dim)
        self.k = torch.nn.Linear(embed_dim, embed_dim)
        self.v = torch.nn.Linear(embed_dim, embed_dim)

    def forward(self,x_q,x_k,x_v, attn_mask):

        Q = self.q(x_q)
        K = self.k(x_k)
        V = self.v(x_v)

        out = self.attn(Q,K,V, attn_mask)
    
        return out


