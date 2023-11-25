import torch

class PositionalEncoding(torch.nn.Module):
    def __init__(self,d_model, max_len = 5000):
        super().__init__()

        pos = torch.arange(max_len).repeat(int(d_model/2),1).T

        #factor in division term
        a = torch.tensor([10000])
        
        #building division term 
        div = torch.arange(0,d_model,2)
        div = ((div/d_model)*(a.log())).exp()

        #performing division and trigonometry
        s = torch.div(pos,div).sin()
        c = torch.div(pos,div).cos()

        #reassembling
        self.pos_embeds = torch.stack([s,c],dim = 2).reshape(max_len,d_model)

    def forward(self,x):
        return x + self.pos_embeds
