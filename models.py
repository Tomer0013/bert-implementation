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
        input_mask = torch.zeros_like(input_ids) == input_ids
        x = self.word_embeddings(input_ids, token_type_ids)
        for enc in self.enc_layers:
            x = enc(x, input_mask)

        return x

    def get_pooled_output(self, input_ids: torch.Tensor, token_type_ids: torch.Tensor) -> torch.Tensor:
        x = self(input_ids, token_type_ids)
        x = self.pooler(x[:, 0:1, :].squeeze())
        x = torch.tanh(x)

        return x


class BertClassifier(nn.Module):
    """
    BERT Model with another linear layer at the end, for the classification task.
    If num_classes = 1, this turns into a regressor (needed for STS-B task).
    
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
        num_classes (int): Number of classes in the classifying task. This is basically the output dim.
    """

    def __init__(self, ckpt_path: str, hidden_size: int, num_layers: int, num_attn_heads: int,
                 intermediate_size: int, num_embeddings: int, max_seq_len: int,
                 drop_prob: float, attn_drop_prob: float, num_classes: int) -> None:
        super(BertClassifier, self).__init__()
        self.drop_prob = drop_prob
        self.bert = BERT(hidden_size=hidden_size, num_layers=num_layers, num_attn_heads=num_attn_heads,
                         intermediate_size=intermediate_size, num_embeddings=num_embeddings,
                         max_seq_len=max_seq_len, drop_prob=drop_prob, attn_drop_prob=attn_drop_prob)
        self.output_layer = nn.Linear(hidden_size, num_classes)

        torch.nn.init.trunc_normal_(self.output_layer.weight, std=0.02)
        self.bert.load_state_dict(create_pretrained_state_dict_from_google_ckpt(ckpt_path))

    def forward(self, input_ids: torch.Tensor, token_type_ids: torch.Tensor) -> torch.Tensor:
        x = self.bert.get_pooled_output(input_ids, token_type_ids)
        x = torch.dropout(x, self.drop_prob, self.training)
        x = self.output_layer(x)
        if x.shape[1] == 1:
            x = x.squeeze()

        return x


class SQuADBert(nn.Module):
    """
    BERT Model for the SQuAD task.

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
    """

    def __init__(self, ckpt_path: str, hidden_size: int, num_layers: int, num_attn_heads: int,
                 intermediate_size: int, num_embeddings: int, max_seq_len: int,
                 drop_prob: float, attn_drop_prob: float) -> None:
        super(SQuADBert, self).__init__()
        self.bert = BERT(hidden_size=hidden_size, num_layers=num_layers, num_attn_heads=num_attn_heads,
                         intermediate_size=intermediate_size, num_embeddings=num_embeddings,
                         max_seq_len=max_seq_len, drop_prob=drop_prob, attn_drop_prob=attn_drop_prob)
        self.p_start = nn.Linear(hidden_size, 1)
        self.p_end = nn.Linear(hidden_size, 1)

        for x in [self.p_start, self.p_end]:
            torch.nn.init.trunc_normal_(x.weight, std=0.02)
        self.bert.load_state_dict(create_pretrained_state_dict_from_google_ckpt(ckpt_path))

    def forward(self, input_ids: torch.Tensor, token_type_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.bert(input_ids, token_type_ids)
        p_start_logits = self.p_start(x).squeeze(dim=-1)
        p_end_logits = self.p_end(x).squeeze(dim=-1)

        return p_start_logits, p_end_logits
