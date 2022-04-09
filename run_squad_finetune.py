import os
import torch

from tqdm import tqdm
from models import SQuADBert
from preprocessing import SQuADOpsHandler
from utils import get_device, set_random_seed
from args import get_squad_args
from torch.nn.functional import cross_entropy


# Get args
args = get_squad_args()
use_squad_v1 = bool(args.use_squad_v1)
do_lower_case = bool(args.do_lower_case)
set_random_seed(args.random_seed)
vocab_path = os.path.join(args.pretrained_model_path, "vocab.txt")
ckpt_path = os.path.join(args.pretrained_model_path, "bert_model.ckpt")

# Init
data_path = os.path.join(args.datasets_path, "squad_data/")
squad_ops_handler = SQuADOpsHandler(args.max_seq_len, args.max_query_len, args.doc_stride, args.max_answer_len,
                                    use_squad_v1, do_lower_case, data_path, vocab_path)
train_dataset = squad_ops_handler.get_train_dataset()
dev_dataset = squad_ops_handler.get_dev_dataset()
device = get_device()
model = SQuADBert(ckpt_path=ckpt_path, hidden_size=768, num_layers=12,
                  num_attn_heads=12, intermediate_size=3072,
                  num_embeddings=30522, max_seq_len=512, drop_prob=0.1,
                  attn_drop_prob=0.1)
model.to(device)
train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=args.batch_size,
                                           shuffle=True,
                                           num_workers=args.num_workers_dataloader)
dev_loader = torch.utils.data.DataLoader(dev_dataset, batch_size=args.batch_size,
                                         num_workers=args.num_workers_dataloader)
optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, eps=args.optimizer_eps, weight_decay=args.l2_wd)
num_train_steps = int((len(train_dataset) * args.epochs) / args.batch_size)
num_warmup_steps = int(num_train_steps * args.warmup_prop)
sched_decay = torch.optim.lr_scheduler.LinearLR(optimizer, total_iters=num_train_steps,
                                                start_factor=1, end_factor=0)
sched_warmup = torch.optim.lr_scheduler.LinearLR(optimizer, total_iters=num_warmup_steps,
                                                 start_factor=1e-10, end_factor=1)
global_step = 0

# Train
for e in range(args.epochs):
    model.train()
    with torch.enable_grad(), tqdm(total=len(train_dataset)) as progress_bar:
        for batch in train_loader:
            input_ids, token_type_ids, start_labels, end_labels = batch
            input_ids = input_ids.to(device)
            token_type_ids = token_type_ids.to(device)
            start_labels = start_labels.to(device)
            end_labels = end_labels.to(device)
            start_logits, end_logits = model(input_ids, token_type_ids)
            loss = cross_entropy(start_logits, start_labels) + cross_entropy(end_logits, end_labels)
            loss_val = loss.item()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0, norm_type=2)
            optimizer.step()
            sched_warmup.step()
            sched_decay.step()
            global_step += 1
            progress_bar.update(args.batch_size)
            progress_bar.set_postfix(epoch=e, train_loss=loss_val)

    # eval
    dev_loss = 0
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
            dev_loss += loss_function(logits, labels, reduction='sum').item()
            if len(logits.shape) == 1:
                preds_list += logits.tolist()
            else:
                preds_list += logits.max(dim=-1)[1].tolist()
            labels_list += labels.tolist()
    print("\n***** Eval results *****")
    for eval_type, eval_func in task_eval_metrics:
        print(f"eval_{eval_type}: {eval_func(preds_list, labels_list):.6f}")
    print(f"eval_loss: {dev_loss / len(dev_dataset):.6f}")
    print(f"global_step: {global_step}")



