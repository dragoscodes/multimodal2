import torch
import torch.nn as nn

class LearnedPositionalEncoding(nn.Module):
    def __init__(self, max_len, d_model):
        super().__init__()
        self.position_embeddings = nn.Parameter(torch.zeros(max_len, d_model))

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, d_model)
        """
        return x + self.position_embeddings[:x.size(1), :]