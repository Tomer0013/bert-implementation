import torch
import torch.nn as nn
import math


# def masked_softmax(logits: torch.Tensor, mask: torch.Tensor, dim: int = -1, log_softmax: bool = False) -> torch.Tensor:
#     """Take the softmax of `logits` over given dimension, and set
#     entries to 0 wherever `mask` is 0.
#
#     Args:
#         logits (torch.Tensor): Inputs to the softmax function.
#         mask (torch.Tensor): Same shape as `logits`, with 0 indicating
#             positions that should be assigned 0 probability in the output.
#         dim (int): Dimension over which to take softmax.
#         log_softmax (bool): Take log-softmax rather than regular softmax.
#             E.g., some PyTorch functions such as `F.nll_loss` expect log-softmax.
#     """
#     mask = mask.type(torch.float32)
#     masked_logits = mask * logits + (1 - mask) * -1e30
#     softmax_fn = nn.functional.log_softmax if log_softmax else nn.functional.softmax
#     probs = softmax_fn(masked_logits, dim)
#
#     return probs
#
#
# class MultiHeadAttention(nn.Module):
#     """
#     Multi headed attention as described in the paper
#     "Attention is All You Need".
#
#     Args:
#         input_size (int): Dim size of input.
#         num_heads (int): Number of heads.
#         attn_drop_prob (float): Dropout probability right after the softmax.
#     """
#     def __init__(self, input_size: int, num_heads: int, attn_drop_prob: float = 0.0) -> None:
#         assert input_size % num_heads == 0, "Input size divided by num heads must be a whole number."
#         super(MultiHeadAttention, self).__init__()
#         self.num_heads = num_heads
#         self.attn_drop_prob = attn_drop_prob
#         self.q_proj = torch.nn.Linear(input_size, input_size)
#         self.k_proj = torch.nn.Linear(input_size, input_size)
#         self.v_proj = torch.nn.Linear(input_size, input_size)
#         self.o_proj = torch.nn.Linear(input_size, input_size)
#
#     def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
#         n, t, d = x.shape
#
#         q = self.q_proj(x).view(n, t, self.num_heads, d // self.num_heads).transpose(1, 2)
#         k = self.k_proj(x).view(n, t, self.num_heads, d // self.num_heads).transpose(1, 2)
#         v = self.v_proj(x).view(n, t, self.num_heads, d // self.num_heads).transpose(1, 2)
#
#         s = q.matmul(k.transpose(-2, -1)) / math.sqrt(d // self.num_heads)
#         s = masked_softmax(s, mask.view(n, 1, 1, -1))
#         s = torch.dropout(s, self.attn_drop_prob, self.training)
#         a = s.matmul(v).transpose(1, 2).contiguous().view(n, t, -1)
#         x = self.o_proj(a)
#
#         return x

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

        q = self.q_proj(x).view(n, t, self.num_heads, d // self.num_heads).transpose(1, 2)
        k = self.k_proj(x).view(n, t, self.num_heads, d // self.num_heads).transpose(1, 2)
        v = self.v_proj(x).view(n, t, self.num_heads, d // self.num_heads).transpose(1, 2)

        s = q.matmul(k.transpose(-2, -1)) / math.sqrt(d // self.num_heads)
        s.masked_fill_(mask.view(n, 1, 1, -1), -torch.inf)
        s = torch.softmax(s, dim=-1)
        s = torch.dropout(s, self.attn_drop_prob, self.training)
        a = s.matmul(v).transpose(1, 2).contiguous().view(n, t, -1)
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


class WordEmbeddings(nn.Module):
    """
    Word embeddings layer cosisting of word embeddings, token type embeddings
    and learned position embeddings.

    Args:
        num_embeddings (int): Number of words in the embedding.
        embedding_dim (int): Embedding dimension size.
        max_seq_len (int): Maximum possible sequence length.
        drop_prob (float): Dropout probability.
    """
    def __init__(self, num_embeddings: int, embedding_dim: int, max_seq_len: int, drop_prob: float) -> None:
        super(WordEmbeddings, self).__init__()
        self.drop_prob = drop_prob
        self.word_embeddings = nn.Embedding(num_embeddings, embedding_dim)
        self.position_embeddings = nn.Parameter(torch.zeros((max_seq_len, embedding_dim)))
        self.token_type_embeddings = nn.Embedding(2, embedding_dim)
        self.layer_norm = nn.LayerNorm(embedding_dim)

        nn.init.xavier_uniform_(self.position_embeddings)

    def forward(self, word_ids: torch.Tensor, token_type_ids: torch.Tensor) -> torch.Tensor:
        w_emb = self.word_embeddings(word_ids)
        t_emb = self.token_type_embeddings(token_type_ids)
        x = w_emb + t_emb + self.position_embeddings[:w_emb.shape[1], :]
        x = self.layer_norm(x)
        x = torch.dropout(x, self.drop_prob, self.training)

        return x
