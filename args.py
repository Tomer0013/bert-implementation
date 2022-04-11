import argparse


def get_glue_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size",
                        type=int,
                        default=32,
                        help="Batch size during training and eval.")
    parser.add_argument("--lr",
                        type=float,
                        default=5e-5,
                        help="Learning rate.")
    parser.add_argument("--l2_wd",
                        type=float,
                        default=1e-2,
                        help="L2 weight decay.")
    parser.add_argument("--epochs",
                        type=int,
                        default=3,
                        help="Number of epochs.")
    parser.add_argument("--max_seq_len",
                        type=int,
                        default=128,
                        help="Maximum sequence length.")
    parser.add_argument("--warmup_prop",
                        type=float,
                        default=0.1,
                        help="Percentage of global steps to be used for lr warm up in the beginning.")
    parser.add_argument("--random_seed",
                        type=int,
                        default=13,
                        help="Random seed.")
    parser.add_argument("--num_workers_dataloader",
                        type=int,
                        default=4,
                        help="Number of workers for data loader.")
    parser.add_argument("--optimizer_eps",
                        type=float,
                        default=1e-6,
                        help="AdamW epsilon.")
    parser.add_argument("--pretrained_model_path",
                        type=str,
                        default="./bert_base_pretrained/uncased_L-12_H-768_A-12",
                        help="Path of the folder containing the google ckpt files and vocab.")
    parser.add_argument("--datasets_path",
                        type=str,
                        default="./datasets/",
                        help="Path of the folder containing the GLUE and SQUAD datasets.")
    parser.add_argument("--task_name",
                        type=str,
                        default=None,
                        help="Type of fine-tuning task.")

    args = parser.parse_args()

    return args


def get_squad_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size",
                        type=int,
                        default=16,
                        help="Batch size during training.")
    parser.add_argument("--dev_batch_size",
                        type=int,
                        default=256,
                        help="Batch size during eval.")
    parser.add_argument("--lr",
                        type=float,
                        default=5e-5,
                        help="Learning rate.")
    parser.add_argument("--l2_wd",
                        type=float,
                        default=1e-2,
                        help="L2 weight decay.")
    parser.add_argument("--epochs",
                        type=int,
                        default=3,
                        help="Number of epochs.")
    parser.add_argument("--max_seq_len",
                        type=int,
                        default=384,
                        help="Maximum sequence length.")
    parser.add_argument("--warmup_prop",
                        type=float,
                        default=0.1,
                        help="Percentage of global steps to be used for lr warm up in the beginning.")
    parser.add_argument("--random_seed",
                        type=int,
                        default=13,
                        help="Random seed.")
    parser.add_argument("--num_workers_dataloader",
                        type=int,
                        default=4,
                        help="Number of workers for data loader.")
    parser.add_argument("--optimizer_eps",
                        type=float,
                        default=1e-6,
                        help="AdamW epsilon.")
    parser.add_argument("--pretrained_model_path",
                        type=str,
                        default="./bert_base_pretrained/uncased_L-12_H-768_A-12",
                        help="Path of the folder containing the google ckpt files and vocab.")
    parser.add_argument("--datasets_path",
                        type=str,
                        default="./datasets/",
                        help="Path of the folder containing the GLUE and SQUAD datasets.")
    parser.add_argument("--max_answer_len",
                        type=int,
                        default=30,
                        help="Maximum sequence length.")
    parser.add_argument("--max_query_len",
                        type=int,
                        default=64,
                        help="Maximum query length.")
    parser.add_argument("--doc_stride",
                        type=int,
                        default=128,
                        help="Window size for sliding window, when going over contexts longer than max length.")
    parser.add_argument("--do_lower_case",
                        type=lambda s: s.lower().startswith("t"),
                        default=True,
                        help="Should be True if model is uncased, False otherwise.")
    parser.add_argument("--use_squad_v1",
                        type=lambda s: s.lower().startswith("t"),
                        default=False,
                        help="Whether to use SQuAD v1 (no questions without an answer) or not.")
    args = parser.parse_args()

    return args
