import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task",
                        type=str,
                        help="Type of fine-tuning task.")
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
                        help="Path of the folder with the GLUE and SQUAD datasets.")
    args = parser.parse_args()

    return args
