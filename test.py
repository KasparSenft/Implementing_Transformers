from dataset import load_dataset
from torch.utils.data import DataLoader
from modelling import Transformer
import torch
import numpy as np


np.random.seed(999)
torch.manual_seed(999)


train_ds, val_ds, test_ds = load_dataset()

# def collate_fn(items):

#     output = {[dic[k] for dic in items] for k in items[0]}

#     return output


args = {
    'vocab_size': 3,
    'd_model': 256,
    'num_heads': 4,
    'num_decoder_layers': 4,
    'dim_feed_forward': 64,
    'dropout': 0.1,
    'max_len': 512,
}

model = Transformer(**args)

input = torch.randint(high = 3, size = (4,6))
mask = torch.rand(size = (4,6)) > 0.9

input_target = torch.randint(high = 3, size = (4,6))
target_mask = torch.rand(size = (4,6)) > 0.9



print((model(input, mask, input_target, target_mask)))




