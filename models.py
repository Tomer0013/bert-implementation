import torch
import torch.nn as nn
import layers

from tensorflow.train import load_checkpoint

class BERT(nn.Module):
    """
    BERT Model as described in
    https://arxiv.org/pdf/1810.04805.pdf

    Args:
        hidden_size (int): Hidden dim size.
        num_layers (int): Number of encoder layers.
        num_attn_heads (int): Number of heads within multihead attention.
        intermediate_size (int): Dim size of first linear layer within the feedforward.
        num_embeddings (int): Number of words in the embedding.
        max_seq_len (int): Maximum possible sequence length.
        drop_prob (float): Dropout probability.
        attn_drop_prob (float): Droput probability within the attention after the softmax.
    """
    def __init__(self, hidden_size: int, num_layers: int, num_attn_heads: int,
                 intermediate_size: int, num_embeddings: int, max_seq_len: int,
                 drop_prob: float, attn_drop_prob: float) -> None:
        super(BERT, self).__init__()
        self.word_embeddings = layers.WordEmbeddings(num_embeddings, hidden_size, max_seq_len)
        self.pos_enc = layers.PositionalEncoding(max_seq_len, hidden_size)
        self.enc_layers = nn.ModuleList(
            [layers.TransformerEncoderBlock(hidden_size, num_attn_heads,
             intermediate_size, drop_prob, attn_drop_prob) for _ in range(num_layers)]
        )

    def forward(self, input_ids):
        pass

