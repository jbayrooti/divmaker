import math
import torch
import numpy as np

from src.utils.utils import l2_normalize

class DivMakerLoss(object):
    
    def __init__(self, outputs_orig, outputs1, outputs2, outputs3=None, t=0.07):
        super().__init__()
        self.outputs1 = l2_normalize(outputs1, dim=1)
        self.outputs2 = l2_normalize(outputs2, dim=1)
        if outputs3 is not None:
            self.outputs3 = l2_normalize(outputs3, dim=1)
        else:
            self.outputs3 = None
        self.outputs_orig = l2_normalize(outputs_orig, dim=1)
        self.t = t

    def get_loss(self):
        # See also https://arxiv.org/pdf/2103.16440.pdf for an application of this loss for anomaly detection

        batch_size = self.outputs_orig.size(0)  # batch_size x out_dim
        
        sim_x_x1 = torch.sum(self.outputs1 * self.outputs_orig, dim=-1) / self.t # [256]
        sim_x_x2 = torch.sum(self.outputs2 * self.outputs_orig, dim=-1) / self.t # [256]
        sim_x1_x2 = torch.sum(self.outputs1 * self.outputs2, dim=-1) / self.t # [256]
        
        sim_x1_12_cat = torch.cat([sim_x_x1.unsqueeze(-1), sim_x1_x2.unsqueeze(-1)], dim=-1) # [256, 2]
        sim_x1_12_norm = torch.logsumexp(sim_x1_12_cat, dim=1) # [256]

        sim_x2_12_cat = torch.cat([sim_x_x2.unsqueeze(-1), sim_x1_x2.unsqueeze(-1)], dim=-1) # [256, 2]
        sim_x2_12_norm = torch.logsumexp(sim_x2_12_cat, dim=1) # [256]
        
        if self.outputs3 is None:
            loss = -torch.mean((sim_x_x1 - sim_x1_12_norm) + (sim_x_x2 - sim_x2_12_norm))
            return loss

        sim_x_x3 = torch.sum(self.outputs3 * self.outputs_orig, dim=-1) / self.t # [256]
        sim_x1_x3 = torch.sum(self.outputs1 * self.outputs3, dim=-1) / self.t # [256]
        sim_x2_x3 = torch.sum(self.outputs2 * self.outputs3, dim=-1) / self.t # [256]

        sim_x1_123_cat = torch.cat([sim_x_x1.unsqueeze(-1), sim_x1_x2.unsqueeze(-1), sim_x1_x3.unsqueeze(-1)], dim=-1) # [256, 2]
        sim_x1_123_norm = torch.logsumexp(sim_x1_123_cat, dim=1) # [256]

        sim_x2_123_cat = torch.cat([sim_x_x2.unsqueeze(-1), sim_x1_x2.unsqueeze(-1), sim_x2_x3.unsqueeze(-1)], dim=-1) # [256, 2]
        sim_x2_123_norm = torch.logsumexp(sim_x2_123_cat, dim=1) # [256]

        sim_x3_123_cat = torch.cat([sim_x_x3.unsqueeze(-1), sim_x1_x3.unsqueeze(-1), sim_x2_x3.unsqueeze(-1)], dim=-1) # [256, 2]
        sim_x3_123_norm = torch.logsumexp(sim_x3_123_cat, dim=1) # [256]

        loss = -torch.mean((sim_x_x1 - sim_x1_123_norm) + (sim_x_x2 - sim_x2_123_norm) + (sim_x_x3 - sim_x1_123_norm))
        return loss