import numpy as np
import tokenization

from utils import read_tsv_file


def truncate_seq_pair(tokens_a: list, tokens_b: list, max_length: int) -> None:
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def prep_sentence_pairs_data(data_path: str, vocab_path: str, label_col: int, text_a_col: int,
                             text_b_col: int, max_seq_len: int, incl_first_row: bool):
    tokenizer = tokenization.FullTokenizer(vocab_path)
    start_idx = 1
    if incl_first_row:
        start_idx = 0
    input_ids_list = []
    token_type_ids_list = []
    labels = []
    raw_train = read_tsv_file(data_path)
    for row in raw_train[start_idx:]:
        label = int(tokenization.convert_to_unicode(row[label_col]))
        text_a = row[text_a_col]
        text_b = row[text_b_col]
        tokens_a = tokenization.convert_to_unicode(text_a)
        tokens_b = tokenization.convert_to_unicode(text_b)
        tokens_a = tokenizer.tokenize(tokens_a)
        tokens_b = tokenizer.tokenize(tokens_b)
        truncate_seq_pair(tokens_a, tokens_b, max_seq_len - 3)
        tokens = ["[CLS]"] + tokens_a + ["[SEP]"] + tokens_b + ["[SEP]"]
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        token_type_ids = [0] * (len(tokens_a) + 2) + [1] * (len(tokens_b) + 1)
        for x in [input_ids, token_type_ids]:
            x.extend([0]*(max_seq_len - len(x)))
        input_ids_list.append(input_ids)
        token_type_ids_list.append(token_type_ids)
        labels.append(label)

    return np.array(input_ids_list), np.array(token_type_ids_list), np.array(labels)
