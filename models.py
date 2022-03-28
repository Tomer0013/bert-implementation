import torch
import torch.nn as nn
import layers

from utils import create_pretrained_state_dict_from_google_ckpt


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

    def forward(self, input_ids: torch.Tensor, token_type_ids: torch.Tensor) -> torch.Tensor:
        input_mask = torch.zeros_like(input_ids) != input_ids
        x = self.word_embeddings(input_ids, token_type_ids)
        for enc in self.enc_layers:
            x = enc(x, input_mask)

        return x

    def get_pooled_output(self, input_ids: torch.Tensor, token_type_ids: torch.Tensor) -> torch.Tensor:
        x = self(input_ids, token_type_ids)
        x = self.pooler(x[:, 0:1, :].squeeze())

        return x


class ClassifierBERT(nn.Module):
    """
    BERT Model with another linear layer at the end, for the classification task.
    
    Args:
        ckpt_path (str): Path to pretrained weights.
        hidden_size (int): Hidden dim size.
        num_layers (int): Number of encoder layers.
        num_attn_heads (int): Number of heads within multihead attention.
        intermediate_size (int): Dim size of first linear layer within the feedforward.
        num_embeddings (int): Number of words in the embedding.
        max_seq_len (int): Maximum possible sequence length.
        drop_prob (float): Dropout probability.
        attn_drop_prob (float): Droput probability within the attention after the softmax.
        num_classes (int): Number of classes in the classifying task.
    """

    def __init__(self, ckpt_path: str, hidden_size: int, num_layers: int, num_attn_heads: int,
                 intermediate_size: int, num_embeddings: int, max_seq_len: int,
                 drop_prob: float, attn_drop_prob: float, num_classes: int) -> None:
        super(ClassifierBERT, self).__init__()
        self.drop_prob = drop_prob
        self.bert = BERT(hidden_size=hidden_size, num_layers=num_layers, num_attn_heads=num_attn_heads,
                         intermediate_size=intermediate_size, num_embeddings=num_embeddings,
                         max_seq_len=max_seq_len, drop_prob=drop_prob, attn_drop_prob=attn_drop_prob)
        self.output_layer = nn.Linear(hidden_size, num_classes)

        self.bert.load_state_dict(create_pretrained_state_dict_from_google_ckpt(ckpt_path))

    def forward(self, input_ids: torch.Tensor, token_type_ids: torch.Tensor) -> torch.Tensor:
        x = self.bert.get_pooled_output(input_ids, token_type_ids)
        x = torch.dropout(x, self.drop_prob, self.training)
        x = self.output_layer(x)
        x = torch.log_softmax(x, dim=-1)

        return x
