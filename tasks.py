import os.path

import numpy as np
import tokenization

from preprocessing import prep_sentence_pairs_data
from task_datasets import GlueDataset


def mrpc_task(data_path: str, vocab_path, max_seq_len: int):
    train_path = os.path.join(data_path, "train.tsv")
    dev_path = os.path.join(data_path, "dev.tsv")

    # Train
    input_ids, token_type_ids, labels = prep_sentence_pairs_data(train_path, vocab_path, 0, 3, 4,
                                                                 max_seq_len, False)
    train_dataset = GlueDataset(input_ids, token_type_ids, labels)

    # Dev
    input_ids, token_type_ids, labels = prep_sentence_pairs_data(dev_path, vocab_path, 0, 3, 4,
                                                                 max_seq_len, False)
    dev_dataset = GlueDataset(input_ids, token_type_ids, labels)

    return train_dataset, dev_dataset





