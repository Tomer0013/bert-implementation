import torch.nn.functional
import torch.utils.data as data

from models import ClassifierBERT
from tasks import mrpc_task
from utils import get_device


device = get_device()
num_workers = 4
batch_size = 32
max_seq_length = 128
data_path = 'datasets/glue_data/MRPC/'
vocab_path = "bert_base_pretrained/uncased_L-12_H-768_A-12/vocab.txt"
ckpt_path = "bert_base_pretrained/uncased_L-12_H-768_A-12/bert_model.ckpt"
model = ClassifierBERT(ckpt_path=ckpt_path, hidden_size=768, num_layers=12, num_attn_heads=12, intermediate_size=3072,
                       num_embeddings=30522, max_seq_len=512, drop_prob=0.1, attn_drop_prob=0.1, num_classes=2)
model.to(device)
train_dataset, dev_dataset = mrpc_task(data_path, vocab_path, max_seq_length)
train_loader = data.DataLoader(train_dataset,
                               batch_size=batch_size,
                               shuffle=True,
                               num_workers=num_workers)
optimizer = torch.optim.Adam(model.parameters())

model.train()
epochs = 1
for e in range(epochs):
    for t, batch in enumerate(train_loader):
        input_ids, token_type_ids, labels = batch
        input_ids = input_ids.to(device)
        token_type_ids = token_type_ids.to(device)
        labels = labels.to(device)
        log_probs = model(input_ids, token_type_ids)
        loss = torch.nn.functional.nll_loss(log_probs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
