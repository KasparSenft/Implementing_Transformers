import torch
import numpy as np
from torch import nn
from torch.optim import Optimizer, AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import Subset
from evaluate import load
import random

class LearningRateScheduler(LambdaLR):

    def __init__(self,
                optimizer: Optimizer,
                d_model: int,
                warmup_steps: int,
                last_epoch = -1):
        
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.num_param_groups = len(optimizer.param_groups)

        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        lr = np.power(self.d_model, -0.5) * np.minimum(np.power(self._step_count, -0.5),self._step_count * np.power(self.warmup_steps, -1.5))
        return [lr] * self.num_param_groups
    
def get_adam_optimizer(model, lr, weight_decay, exclude_params=['bias', 'LayerNorm']):
    parameters = [
        {'params': [p for n, p in model.named_parameters() if n not in exclude_params], 'weight_decay': weight_decay},
        {'params': [p for n, p in model.named_parameters() if n in exclude_params], 'weight_decay': 0.0}  # No weight decay for certain parameters
    ]
    optimizer = AdamW(parameters, lr=lr)
    return optimizer
    

def translation_collate_fn(dict_list):
    batch = {'en':[],'de':[]}

    for lang in ['en','de']:
        for entry in dict_list:
            batch[lang].append(entry[lang])

        batch[lang] = torch.tensor(batch[lang]).transpose(1,0)
        
    return batch
    

def get_subset_dataset(dataset, pct):
        subset_len = int(len(dataset)*pct)
        sub_indices = random.sample(range(len(dataset)), subset_len)
        sub_dataset = Subset(dataset, sub_indices)
        return sub_dataset

def evaluate_bleu_score(generator, dataset, src):
     
    trgt = 'en' if src == 'de' else 'de'
    bleu = load('bleu')

     #format dataset into dictionary of lists
    predictions = [generator.generate(entry[src])[0] for entry in dataset]
    references = [[entry[trgt]] for entry in dataset]

    return bleu.compute(predictions = predictions, references = references)
