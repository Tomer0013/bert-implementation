import csv
import numpy as np
import random
import torch
import torch.nn.functional as F

from tensorflow.train import load_checkpoint


def masked_softmax(logits, mask, dim=-1, log_softmax=False):
    """Take the softmax of `logits` over given dimension, and set
    entries to 0 wherever `mask` is 0.

    Args:
        logits (torch.Tensor): Inputs to the softmax function.
        mask (torch.Tensor): Same shape as `logits`, with 0 indicating
            positions that should be assigned 0 probability in the output.
        dim (int): Dimension over which to take softmax.
        log_softmax (bool): Take log-softmax rather than regular softmax.
            E.g., some PyTorch functions such as `F.nll_loss` expect log-softmax.

    Returns:
        probs (torch.Tensor): Result of taking masked softmax over the logits.
    """
    mask = mask.type(torch.float32)
    masked_logits = mask * logits + (1 - mask) * -1e30
    softmax_fn = F.log_softmax if log_softmax else F.softmax
    probs = softmax_fn(masked_logits, dim)

    return probs


def create_pretrained_state_dict_from_google_ckpt(ckpt_path: str) -> dict:
    reader = load_checkpoint(ckpt_path)
    pretrained_state_dict = {
        v: reader.get_tensor(v) for v in reader.get_variable_to_shape_map()
    }
    convert_dict = {
        "attention/self/key/kernel": "attn_res_block.func.k_proj.weight",
        "intermediate/dense/bias": "ff_res_block.func.lin_1.bias",
        "attention/self/key/bias": "attn_res_block.func.k_proj.bias",
        "attention/output/LayerNorm/gamma": "attn_res_block.layer_norm.weight",
        "attention/output/LayerNorm/beta": "attn_res_block.layer_norm.bias",
        "attention/self/query/bias": "attn_res_block.func.q_proj.bias",
        "attention/output/dense/kernel": "attn_res_block.func.o_proj.weight",
        "attention/self/value/kernel": "attn_res_block.func.v_proj.weight",
        "attention/self/query/kernel": "attn_res_block.func.q_proj.weight",
        "attention/self/value/bias": "attn_res_block.func.v_proj.bias",
        "intermediate/dense/kernel": "ff_res_block.func.lin_1.weight",
        "output/LayerNorm/gamma": "ff_res_block.layer_norm.weight",
        "output/LayerNorm/beta": "ff_res_block.layer_norm.bias",
        "output/dense/kernel": "ff_res_block.func.lin_2.weight",
        "output/dense/bias": "ff_res_block.func.lin_2.bias",
        "attention/output/dense/bias": "attn_res_block.func.o_proj.bias"
    }
    new_state_dict = {}

    # Encoder weights
    for key in pretrained_state_dict.keys():
        if "encoder" in key:
            key_split = key.split("/")
            layer_num = int(key_split[2].split("_")[1])
            key_prefix = f"bert/encoder/layer_{layer_num}/"
            new_key_prefix = f"enc_layers.{layer_num}."
            sub_key = "/".join(key_split[3:])
            if len(pretrained_state_dict[key_prefix + sub_key].shape) == 2:
                new_state_dict[new_key_prefix + convert_dict[sub_key]] = torch.Tensor(
                    pretrained_state_dict[key_prefix + sub_key].T)
            else:
                new_state_dict[new_key_prefix + convert_dict[sub_key]] = torch.Tensor(
                    pretrained_state_dict[key_prefix + sub_key])

    # Embedding weights
    new_state_dict_emb_keys = [
        "word_embeddings.position_embeddings",
        "word_embeddings.word_embeddings.weight",
        "word_embeddings.token_type_embeddings.weight",
        "word_embeddings.layer_norm.weight",
        "word_embeddings.layer_norm.bias"
    ]
    pretrained_state_dict_emb_keys = [
        "bert/embeddings/position_embeddings",
        "bert/embeddings/word_embeddings",
        "bert/embeddings/token_type_embeddings",
        "bert/embeddings/LayerNorm/gamma",
        "bert/embeddings/LayerNorm/beta"
    ]
    for k1, k2 in zip(new_state_dict_emb_keys, pretrained_state_dict_emb_keys):
        new_state_dict[k1] = torch.Tensor(pretrained_state_dict[k2])

    # Pooler weights
    new_state_dict["pooler.weight"] = torch.Tensor(pretrained_state_dict["bert/pooler/dense/kernel"].T)
    new_state_dict["pooler.bias"] = torch.Tensor(pretrained_state_dict["bert/pooler/dense/bias"])

    return new_state_dict


def read_tsv_file(path: str, quotechar: str = None) -> list:
    with open(path, 'r') as f:
        reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
        lines = []
        for line in reader:
            lines.append(line)

        return lines


def get_device() -> str:
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'

    return device


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
