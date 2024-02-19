from torch import nn
import numpy as np
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

class LearningRateScheduler(LRScheduler):

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
    
