import torch
import torch.nn as nn
import layers


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
        self.word_embeddings = layers.WordEmbeddings(num_embeddings, hidden_size, max_seq_len, drop_prob)
        self.enc_layers = nn.ModuleList(
            [layers.TransformerEncoderBlock(hidden_size, num_attn_heads,
             intermediate_size, drop_prob, attn_drop_prob) for _ in range(num_layers)]
        )
        self.pooler = nn.Linear(hidden_size, hidden_size)

    def forward(self, input_ids: torch.Tensor, input_mask: torch.Tensor,
                token_type_ids: torch.Tensor) -> torch.Tensor:
        x = self.word_embeddings(input_ids, token_type_ids)
        for enc in self.enc_layers:
            x = enc(x, input_mask)

        return x

    def get_pooled_output(self, input_ids: torch.Tensor, input_mask: torch.Tensor,
                          token_type_ids: torch.Tensor) -> torch.Tensor:
        x = self(input_ids, input_mask, token_type_ids)
        x = self.pooler(x[:, 0:1, :].squeeze())

        return x

