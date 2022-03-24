from models import BERT

model = BERT(hidden_size=768, num_layers=12, num_attn_heads=12, intermediate_size=3072,
             emb_n_words=30522, max_seq_len=512, drop_prob=0.1, attn_drop_prob=0.1)

model_state_dict = model.state_dict()