import torch
import torch.nn as nn
import math

from utils import masked_softmax


class MultiHeadAttention(nn.Module):
    """
    Multi headed attention as described in the paper
    "Attention is All You Need".

    Args:
        input_size (int): Dim size of input.
        num_heads (int): Number of heads.
        attn_drop_prob (float): Dropout probability right after the softmax.
    """
    def __init__(self, input_size: int, num_heads: int, attn_drop_prob: float = 0.0) -> None:
        assert input_size % num_heads == 0, "Input size divided by num heads must be a whole number."
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.attn_drop_prob = attn_drop_prob
        self.q_proj = torch.nn.Linear(input_size, input_size)
        self.k_proj = torch.nn.Linear(input_size, input_size)
        self.v_proj = torch.nn.Linear(input_size, input_size)
        self.o_proj = torch.nn.Linear(input_size, input_size)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        n, t, d = x.shape

        q = self.q_proj(x).view(n, self.num_heads, t, d // self.num_heads)
        k = self.k_proj(x).view(n, self.num_heads, t, d // self.num_heads)
        v = self.v_proj(x).view(n, self.num_heads, t, d // self.num_heads)

        s = q.matmul(k.transpose(-2, -1)) / math.sqrt(d // self.num_heads)
        s = masked_softmax(s, mask.view(n, 1, 1, -1))
        s = torch.dropout(s, self.attn_drop_prob, self.training)
        a = s.matmul(v).view(n, t, -1)
        x = self.o_proj(a)

        return x


class FeedforwardLayer(nn.Module):
    """
    Feedforward layer of the transformer.

    Args:
        input_size (int): Input dim size.
        intermediate_size (int): Dim size of the fist linear transformation.
        hidden_size (int): Dim size of the output, after the second linear transformation.
    """
    def __init__(self, input_size: int, intermediate_size: int, hidden_size: int) -> None:
        super(FeedforwardLayer, self).__init__()
        self.lin_1 = nn.Linear(input_size, intermediate_size)
        self.lin_2 = nn.Linear(intermediate_size, hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.lin_1(x)
        x = nn.functional.gelu(x)
        x = self.lin_2(x)

        return x


class ResidualBlock(nn.Module):
    """
    Residual block within a transformer encoder. Can contain
    either a multihead attention function or a feedforward function.

    Args:
        input_size (int): Input dim size.
        func (str): Residual block function type. Can be either 'multihead_attn' or 'ff'.
        drop_prob (float): Dropout probability.
    """
    def __init__(self, input_size: int, func: str, drop_prob: float = 0.0, **kwargs) -> None:
        super(ResidualBlock, self).__init__()
        assert func in ("multihead_attn", "ff"), "func can only take 'multihead_attn' or 'ff'"
        self.input_size = input_size
        self.drop_prob = drop_prob
        self.layer_norm = nn.LayerNorm(input_size)
        if func == "multihead_attn":
            self.func = MultiHeadAttention(input_size, **kwargs)
        elif func == "ff":
            self.func = FeedforwardLayer(input_size, **kwargs)

    def forward(self, x: torch.Tensor, *args) -> torch.Tensor:
        x = self.layer_norm(x + torch.dropout(self.func(x, *args), self.drop_prob, self.training))

        return x


class TransformerEncoderBlock(nn.Module):
    """
    Transformer encoder block.

    Args:
        hidden_size (int): Hidden size dim of the transformer.
        num_attn_heads (int): Number of heads for multiheaded attention.
        intermediate_size (int): Dim size of first linear transformation in the ff layer.
        drop_prob (float): Dropout probability after applying the residual block's function.
        attn_drop_prob (float): Dropout proability after the softmax, within the multiheaded attention.
    """
    def __init__(self, hidden_size: int, num_attn_heads: int, intermediate_size: int,
                 drop_prob: float, attn_drop_prob: float) -> None:
        super(TransformerEncoderBlock, self).__init__()
        self.attn_res_block = ResidualBlock(input_size=hidden_size, func='multihead_attn', drop_prob=drop_prob,
                                            num_heads=num_attn_heads, attn_drop_prob=attn_drop_prob)
        self.ff_res_block = ResidualBlock(input_size=hidden_size, func='ff', drop_prob=drop_prob,
                                          intermediate_size=intermediate_size, hidden_size=hidden_size)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = self.attn_res_block(x, mask)
        x = self.ff_res_block(x)

        return x


class PositionalEncoding(nn.Module):
    """
    A module which adds positional encoding to the input.

    Args:
        max_seq_len (int): Maximum possible sequence length supported.
        dim (int): Input dimension.
    """
    def __init__(self, max_seq_len, dim):
        super(PositionalEncoding, self).__init__()
        idx_i = torch.arange(max_seq_len, dtype=torch.float32)
        idx_j = torch.arange(dim, dtype=torch.float32)
        grid_i, grid_j = torch.meshgrid(idx_i, idx_j, indexing='ij')
        p = torch.where(grid_j % 2 == 0,
                        torch.sin(grid_i * 10000 ** (-grid_j / dim)),
                        torch.cos(grid_i * 10000 ** (-(grid_j - 1) / dim)))
        self.register_buffer('p', p)

    def forward(self, x):
        x = x + self.p[:x.shape[1], :]

        return x




