import os.path

import numpy as np
import tokenization

from preprocessing import prep_sentence_pairs_data
from task_datasets import GlueDataset
from utils import read_tsv_file


def mrpc_task(data_path: str, vocab_path, max_seq_len: int):
    raw_train = read_tsv_file(os.path.join(data_path, "train.tsv"))
    raw_dev = read_tsv_file(os.path.join(data_path, "dev.tsv"))

    datasets = []
    for raw_data in [raw_train, raw_dev]:
        data = []
        for row in raw_data[1:]:
            label = tokenization.convert_to_unicode(row[0])
            text_a = tokenization.convert_to_unicode(row[3])
            text_b = tokenization.convert_to_unicode(row[4])
            data.append([label, text_a, text_b])
        input_ids, token_type_ids, labels = prep_sentence_pairs_data(data, vocab_path, max_seq_len)
        datasets.append(GlueDataset(input_ids, token_type_ids, labels))

    train_dataset, dev_dataset = datasets

    return train_dataset, dev_dataset





