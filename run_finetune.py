import numpy as np
import torch.nn as nn
import torch.nn.functional
import torch.utils.data as data

from tqdm import tqdm
from models import BertClassifier
from tasks import mrpc_task
from utils import get_device, set_random_seed


set_random_seed(13)
device = get_device()
num_workers = 4
batch_size = 32
max_seq_length = 128
warmup_proportion = 0.1
data_path = 'datasets/glue_data/MRPC/'
vocab_path = "bert_base_pretrained/uncased_L-12_H-768_A-12/vocab.txt"
ckpt_path = "bert_base_pretrained/uncased_L-12_H-768_A-12/bert_model.ckpt"
model = BertClassifier(ckpt_path=ckpt_path, hidden_size=768, num_layers=12, num_attn_heads=12, intermediate_size=3072,
                       num_embeddings=30522, max_seq_len=512, drop_prob=0.1, attn_drop_prob=0.1, num_classes=2)
model.to(device)
train_dataset, dev_dataset = mrpc_task(data_path, vocab_path, max_seq_length)
train_loader = data.DataLoader(train_dataset,
                               batch_size=batch_size,
                               shuffle=True,
                               num_workers=num_workers)
dev_loader = data.DataLoader(dev_dataset, batch_size=batch_size, num_workers=num_workers)
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, eps=1e-6, weight_decay=1e-2)


epochs = 3
num_train_steps = int((len(train_dataset) * epochs) / batch_size)
num_warmup_steps = int(num_train_steps * warmup_proportion)
sched_decay = torch.optim.lr_scheduler.LinearLR(optimizer, total_iters=num_train_steps,
                                                start_factor=1, end_factor=0)
sched_warmup = torch.optim.lr_scheduler.LinearLR(optimizer, total_iters=num_warmup_steps,
                                                 start_factor=1e-10, end_factor=1)

for e in range(epochs):
    model.train()
    with torch.enable_grad(), tqdm(total=len(train_dataset)) as progress_bar:
        for t, batch in enumerate(train_loader):
            input_ids, token_type_ids, labels = batch
            input_ids = input_ids.to(device)
            token_type_ids = token_type_ids.to(device)
            labels = labels.to(device)
            logits = model(input_ids, token_type_ids)
            log_probs = torch.log_softmax(logits, dim=-1)
            loss = torch.nn.functional.nll_loss(log_probs, labels)
            loss_val = loss.item()
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0, norm_type=2)
            optimizer.step()
            sched_warmup.step()
            sched_decay.step()
            progress_bar.update(batch_size)
            progress_bar.set_postfix(epoch=e, NLL=loss_val)

    # eval
    num_correct = 0
    num_samples = 0
    acc_loss = []
    model.eval()
    with torch.no_grad():
        for batch in dev_loader:
            input_ids, token_type_ids, labels = batch
            input_ids = input_ids.to(device)
            token_type_ids = token_type_ids.to(device)
            labels = labels.to(device)
            logits = model(input_ids, token_type_ids)
            log_probs = torch.log_softmax(logits, dim=-1)
            acc_loss.append(torch.nn.functional.nll_loss(log_probs, labels).item())
            _, preds = logits.max(-1)
            num_correct += (preds == labels).sum().item()
            num_samples += len(labels)
    print(f"Dev set accuracy: {num_correct / num_samples}")
    print(f"Dev set nll loss: {np.mean(acc_loss)}")



