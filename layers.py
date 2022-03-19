import torch
import torch.nn as nn
import math


class MultiHeadAttention(nn.Module):
    """
    Multi headed attention as described in the paper
    "Attention is All You Need".

    Args:
        input_size (int): Dim size of input.
        num_heads (int): Number of heads.
    """
    def __init__(self, input_size, num_heads, drop_prob=0):
        assert input_size % num_heads == 0, "Input size divided by num heads must be a whole number."
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.drop_prob = drop_prob
        self.q_proj = torch.nn.Linear(input_size, input_size)
        self.k_proj = torch.nn.Linear(input_size, input_size)
        self.v_proj = torch.nn.Linear(input_size, input_size)
        self.o_proj = torch.nn.Linear(input_size, input_size)

    def forward(self, x, mask):
        n, t, d = x.shape

        q = self.q_proj(x).view(n, self.num_heads, t, d // self.num_heads)
        k = self.k_proj(x).view(n, self.num_heads, t, d // self.num_heads)
        v = self.v_proj(x).view(n, self.num_heads, t, d // self.num_heads)

        s = q.matmul(k.transpose(-2, -1)) / math.sqrt(d // self.num_heads)
        s = masked_softmax(s, mask.view(n, 1, 1, -1))
        s = torch.dropout(s, self.drop_prob, self.training)
        a = s.matmul(v).view(n, t, -1)
        x = self.o_proj(a)

        return x