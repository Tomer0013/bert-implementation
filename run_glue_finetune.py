import argparse
import os
import torch

from tqdm import tqdm
from models import BertClassifier
from tasks import get_task_items
from utils import get_device, set_random_seed
from args import get_glue_args


def main(args: argparse.Namespace) -> None:

    # Get args
    set_random_seed(args.random_seed)
    vocab_path = os.path.join(args.pretrained_model_path, "vocab.txt")
    ckpt_path = os.path.join(args.pretrained_model_path, "bert_model.ckpt")

    # Init
    print("Loading and preprocessing data...")
    num_classes, train_dataset, dev_dataset, \
        task_eval_metrics, loss_function = get_task_items(args.task_name, args.datasets_path, vocab_path, args.max_seq_len)
    device = get_device()
    model = BertClassifier(ckpt_path=ckpt_path, hidden_size=768, num_layers=12,
                           num_attn_heads=12, intermediate_size=3072,
                           num_embeddings=30522, max_seq_len=512, drop_prob=0.1,
                           attn_drop_prob=0.1, num_classes=num_classes)
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
                input_ids, token_type_ids, labels = batch
                input_ids = input_ids.to(device)
                token_type_ids = token_type_ids.to(device)
                labels = labels.to(device)
                logits = model(input_ids, token_type_ids)
                loss = loss_function(logits, labels)
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
        print("\nEvaluating...")
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
        print("***** Eval results *****")
        for eval_type, eval_func in task_eval_metrics:
            print(f"eval_{eval_type}: {eval_func(preds_list, labels_list):.6f}")
        print(f"eval_loss: {dev_loss / len(dev_dataset):.6f}")
        print(f"global_step: {global_step}")


if __name__ == "__main__":
    main(get_glue_args())
