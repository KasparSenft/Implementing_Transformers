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
            attn_logits.masked_fill_(forward_mask == 0, -10e-12)
        
        attn_mask = attn_mask.unsqueeze(dim=1)
        attn_logits.masked_fill_(attn_mask == 0, -10e-12)

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

        self.linear1 = torch.nn.Linear(d_model, dff)
        self.linear2 = torch.nn.Linear(dff, d_model)
        self.relu = torch.nn.ReLU()


    def forward(self, x):
        out = self.linear2(self.relu(self.linear1(x)))
        return out
    

class MultiHeadAttention(torch.nn.Module):
    def __init__(self, d_model, h, mask_future = False):
        super().__init__()

        self.h = h
        self.attn = Attention(mask_future=mask_future)
        self.query_transform = torch.nn.Linear(d_model, d_model, bias=False)
        self.key_transform = torch.nn.Linear(d_model, d_model, bias = False)
        self.value_transform = torch.nn.Linear(d_model, d_model, bias = False)
        self.output_transform = torch.nn.Linear(d_model, d_model, bias = False)


    def forward(self,x_q,x_k,x_v, attn_mask):

        Q = self.transpose_for_mha(self.query_transform(x_q))
        K = self.transpose_for_mha(self.key_transform(x_k))
        V = self.transpose_for_mha(self.value_transform(x_v))

        attn_mask = torch.repeat_interleave(attn_mask, repeats=self.h, dim=0)

        out = self.attn(Q,K,V,attn_mask)
        out = self.untranspose_for_mha(out)
        out = self.output_transform(out)

        return out


    def transpose_for_mha(self, x):
        """"
        args:
            x: torch.tensor, shape = (batch_size, n, d_model*h)
        
        return:
            x: torch.tensor, shape = (batch_size*h, n, d_model)
        """
        x = x.reshape(x.shape[0], x.shape[1], self.h, -1)
        x = x.permute(0,2,1,3)
        x = x.reshape(-1,x.shape[2],x.shape[3])

        return x
    
    def untranspose_for_mha(self,x):
        """
        args:
            x: torch.tensor, shape = (batch_size*h, n, d_model)

        returns:
            x: torch.tensor, shape = (batch_size, n, d_model*h)
        """
        x = x.reshape(-1,self.h,x.shape[1],x.shape[2])
        x = x.permute(0,2,1,3)
        x = x.reshape(x.shape[0], x.shape[1], -1)


        return x
            



class MultiHeadSelfAttention(torch.nn.Module):
    def __init__(self, d_model, h, mask_future = False):
        super().__init__()
        self.MHA = MultiHeadAttention(d_model,h,mask_future)

    def forward(self, x, attn_mask):
        out = self.MHA(x,x,x,attn_mask)
        return out