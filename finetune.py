from models import BERT
from utils import create_pretrained_state_dict_from_google_ckpt
from datasets import load_dataset


model = BERT(hidden_size=768, num_layers=12, num_attn_heads=12, intermediate_size=3072,
             num_embeddings=30522, max_seq_len=512, drop_prob=0.1, attn_drop_prob=0.1)

ckpt_path = "bert_base_pretrained/uncased_L-12_H-768_A-12/bert_model.ckpt"
model.load_state_dict(create_pretrained_state_dict_from_google_ckpt(ckpt_path))


