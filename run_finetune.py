import os.path

import numpy as np
import torch.nn as nn
import torch.nn.functional
import torch.utils.data as data

from tqdm import tqdm
from models import BertClassifier
from tasks import get_task_items
from utils import get_device, set_random_seed
from args import get_args

# Get args
args = get_args()
task_name = args.task
set_random_seed(args.random_seed)
num_workers = args.num_workers_dataloader
batch_size = args.batch_size
max_seq_len = args.max_seq_len
warmup_prop = args.warmup_prop
lr = args.lr
l2_wd = args.l2_wd
opt_eps = args.optimizer_eps
epochs = args.epochs
datasets_path = args.datasets_path
vocab_path = os.path.join(args.pretrained_model_path, "vocab.txt")
ckpt_path = os.path.join(args.pretrained_model_path, "bert_model.ckpt")

# Init
device = get_device()
model = BertClassifier(ckpt_path=ckpt_path, hidden_size=768, num_layers=12, num_attn_heads=12, intermediate_size=3072,
                       num_embeddings=30522, max_seq_len=512, drop_prob=0.1, attn_drop_prob=0.1, num_classes=2)
model.to(device)
train_dataset, dev_dataset, task_eval_metrics = get_task_items(task_name, datasets_path, vocab_path, max_seq_len)
train_loader = data.DataLoader(train_dataset,
                               batch_size=batch_size,
                               shuffle=True,
                               num_workers=num_workers)
dev_loader = data.DataLoader(dev_dataset, batch_size=batch_size, num_workers=num_workers)
optimizer = torch.optim.AdamW(model.parameters(), lr=lr, eps=opt_eps, weight_decay=l2_wd)
num_train_steps = int((len(train_dataset) * epochs) / batch_size)
num_warmup_steps = int(num_train_steps * warmup_prop)
sched_decay = torch.optim.lr_scheduler.LinearLR(optimizer, total_iters=num_train_steps,
                                                start_factor=1, end_factor=0)
sched_warmup = torch.optim.lr_scheduler.LinearLR(optimizer, total_iters=num_warmup_steps,
                                                 start_factor=1e-10, end_factor=1)
global_step = 0

# Train
for e in range(epochs):
    model.train()
    with torch.enable_grad(), tqdm(total=len(train_dataset)) as progress_bar:
        for batch in train_loader:
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
            global_step += 1
            progress_bar.update(batch_size)
            progress_bar.set_postfix(epoch=e, NLL=loss_val)

    # eval
    acc_loss = 0
    preds_list = []
    labels_list = []
    model.eval()
    with torch.no_grad():
        for batch in dev_loader:
            input_ids, token_type_ids, labels = batch
            input_ids = input_ids.to(device)
            token_type_ids = token_type_ids.to(device)
            labels = labels.to(device)
            logits = model(input_ids, token_type_ids)
            log_probs = torch.log_softmax(logits, dim=-1)
            acc_loss += torch.nn.functional.nll_loss(log_probs, labels, reduction='sum').item()
            preds_list += logits.max(dim=-1)[1].tolist()
            labels_list += labels.tolist()
    print("\n***** Eval results *****")
    for eval_type, eval_func in task_eval_metrics:
        print(f"eval_{eval_type}: {eval_func(preds_list, labels_list):.6f}")
    print(f"eval_loss: {acc_loss / len(dev_dataset):.6f}")
    print(f"global_step: {global_step}")



