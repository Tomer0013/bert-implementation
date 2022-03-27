from models import ClassifierBERT


ckpt_path = "bert_base_pretrained/uncased_L-12_H-768_A-12/bert_model.ckpt"
model = ClassifierBERT(ckpt_path=ckpt_path, hidden_size=768, num_layers=12, num_attn_heads=12, intermediate_size=3072,
                       num_embeddings=30522, max_seq_len=512, drop_prob=0.1, attn_drop_prob=0.1, num_classes=2)


input_ids, token_type_ids, labels = dataset[:32]

output = model(input_ids, token_type_ids)