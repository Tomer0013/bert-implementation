import numpy as np
import tokenization

from preprocessing import prep_sentence_pairs_data
from task_datasets import GlueDataset


def mrpc_task(data_path: str, vocab_path, max_seq_len: int):

    data_path = 'datasets/glue_data/MRPC/train.tsv'
    vocab_path = "bert_base_pretrained/uncased_L-12_H-768_A-12/vocab.txt"

    input_ids, token_type_ids, labels = prep_sentence_pairs_data(data_path, vocab_path, 0, 3, 4, max_seq_len, False)
    dataset = GlueDataset(input_ids, token_type_ids, labels)







