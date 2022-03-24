from models import BERT
from utils import create_pytorch_pretrained_state_dict_from_google_ckpt


model = BERT(hidden_size=768, num_layers=12, num_attn_heads=12, intermediate_size=3072,
             num_embeddings=30522, max_seq_len=512, drop_prob=0.1, attn_drop_prob=0.1)

ckpt_path = "bert_base_pretrained/uncased_L-12_H-768_A-12/bert_model.ckpt"
pretrained_state_dict = create_pytorch_pretrained_state_dict_from_google_ckpt(ckpt_path)
model.load_state_dict(pretrained_state_dict)

g = model.state_dict()