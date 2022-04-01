import numpy as np
import tokenization


def truncate_seq_pair(tokens_a: list, tokens_b: list, max_length: int) -> None:
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def prep_sentence_pairs_data(data: list, vocab_path: str, max_seq_len: int) -> tuple:
    tokenizer = tokenization.FullTokenizer(vocab_path)
    input_ids_list = []
    token_type_ids_list = []
    labels = []
    for row in data:
        label = int(row[0])
        text_a = row[1]
        text_b = row[2]
        tokens_a = tokenizer.tokenize(text_a)
        tokens_b = tokenizer.tokenize(text_b)
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
