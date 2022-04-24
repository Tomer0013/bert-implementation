import argparse
import os
import torch

from tqdm import tqdm
from models import SQuADBert
from preprocessing import SQuADOpsHandler
from utils import get_device, set_random_seed
from args import get_squad_args
from torch.nn.functional import cross_entropy
from metrics import squad_compute_em, squad_compute_f1, squad_compute_metric_for_eval


def main(args: argparse.Namespace) -> None:
    # Get args
    set_random_seed(args.random_seed)
    vocab_path = os.path.join(args.pretrained_model_path, "vocab.txt")
    ckpt_path = os.path.join(args.pretrained_model_path, "bert_model.ckpt")

    # Init
    print("Loading and preprocessing data...")
    data_path = os.path.join(args.datasets_path, "squad_data/")
    squad_ops_handler = SQuADOpsHandler(args.max_seq_len, args.max_query_len, args.doc_stride, args.max_answer_len,
                                        args.use_squad_v1, args.do_lower_case, data_path, vocab_path)
    train_dataset = squad_ops_handler.get_train_dataset()
    dev_dataset, eval_items = squad_ops_handler.get_dev_dataset_and_eval_items()
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
    dev_loader = torch.utils.data.DataLoader(dev_dataset,
                                             batch_size=args.dev_batch_size,
                                             shuffle=False,
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
        print("\nEvaluating...")
        dev_loss = 0
        pred_indices_list = []
        model.eval()
        with torch.no_grad():
            for batch_num, batch in enumerate(dev_loader):
                input_ids, token_type_ids, start_labels, end_labels = batch
                input_ids = input_ids.to(device)
                token_type_ids = token_type_ids.to(device)
                start_labels = start_labels.to(device)
                end_labels = end_labels.to(device)
                start_logits, end_logits = model(input_ids, token_type_ids)
                dev_loss += (cross_entropy(start_logits, start_labels, reduction='sum') +
                             cross_entropy(end_logits, end_labels, reduction='sum')).item()
                rel_eval_items = eval_items[batch_num * args.dev_batch_size: (batch_num + 1) * args.dev_batch_size]
                pred_indices_list += squad_ops_handler.logits_to_pred_indices(start_logits, end_logits, rel_eval_items)
        pred_answers = squad_ops_handler.pred_indices_to_final_answers(pred_indices_list, eval_items)
        em_score = squad_compute_metric_for_eval(squad_compute_em, pred_answers, eval_items)
        f1_score = squad_compute_metric_for_eval(squad_compute_f1, pred_answers, eval_items)
        print("***** Eval results *****")
        print(f"eval_f1: {f1_score:.6f}")
        print(f"eval_em: {em_score:.6f}")
        print(f"eval_loss: {dev_loss / len(dev_dataset):.6f}")
        print(f"global_step: {global_step}")


if __name__ == "__main__":
    main(get_squad_args())
