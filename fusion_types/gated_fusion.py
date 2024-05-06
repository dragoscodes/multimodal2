import torch
import torch.nn as nn

class GatedMultimodalUnit(nn.Module):
    def __init__(self, input_dim_v, input_dim_t, hidden_dim):
        super().__init__()
        self.Wv = nn.Linear(input_dim_v, hidden_dim)
        self.Wt = nn.Linear(input_dim_t, hidden_dim)
        self.Wz = nn.Linear(input_dim_v + input_dim_t, hidden_dim)  # Concatenation in input

    def forward(self, xv, xt):
        hv = torch.tanh(self.Wv(xv))
        ht = torch.tanh(self.Wt(xt))
        z = torch.sigmoid(self.Wz(torch.cat([xv, xt], dim=1)))
        h = z * hv + (1 - z) * ht
        return h
