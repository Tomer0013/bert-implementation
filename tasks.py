import os
import tokenization

from preprocessing import prep_sentence_pairs_data, prep_single_sentence_data
from task_datasets import GlueDataset
from utils import read_tsv_file
from metrics import accuracy, f1_score


def get_task_items(task_name: str, datasets_path: str, vocab_path: str, max_seq_len: int) -> tuple:
    assert task_name is not None, "Enter task_name."
    task = task_name.lower()
    assert task in ["mrpc", "mnli", "cola"], f"Invalid task_name: {task}"

    if task == "mrpc":
        return mrpc_task(datasets_path, vocab_path, max_seq_len)

    elif task == "mnli":
        return mnli_task(datasets_path, vocab_path, max_seq_len)

    elif task == "cola":
        return cola_task(datasets_path, vocab_path, max_seq_len)


def mrpc_task(datasets_path: str, vocab_path: str, max_seq_len: int) -> tuple:
    data_path = os.path.join(datasets_path, "glue_data/MRPC/")
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
    eval_metrics = [('accuracy', accuracy), ('f1_score', f1_score)]
    num_classes = 2

    return num_classes, train_dataset, dev_dataset, eval_metrics


def mnli_task(datasets_path: str, vocab_path: str, max_seq_len: int) -> tuple:
    data_path = os.path.join(datasets_path, "glue_data/MNLI/")
    raw_train = read_tsv_file(os.path.join(data_path, "train.tsv"))
    raw_dev = read_tsv_file(os.path.join(data_path, "dev_matched.tsv"))

    label_str_to_int_dict = {"contradiction": 0, "entailment": 1, "neutral": 2}

    datasets = []
    for raw_data in [raw_train, raw_dev]:
        data = []
        for row in raw_data[1:]:
            label = label_str_to_int_dict[tokenization.convert_to_unicode(row[-1])]
            text_a = tokenization.convert_to_unicode(row[8])
            text_b = tokenization.convert_to_unicode(row[9])
            data.append([label, text_a, text_b])
        input_ids, token_type_ids, labels = prep_sentence_pairs_data(data, vocab_path, max_seq_len)
        datasets.append(GlueDataset(input_ids, token_type_ids, labels))

    train_dataset, dev_dataset = datasets
    eval_metrics = [('accuracy', accuracy)]
    num_classes = 3

    return num_classes, train_dataset, dev_dataset, eval_metrics


def cola_task(datasets_path: str, vocab_path: str, max_seq_len: int) -> tuple:
    data_path = os.path.join(datasets_path, "glue_data/CoLA/")
    raw_train = read_tsv_file(os.path.join(data_path, "train.tsv"))
    raw_dev = read_tsv_file(os.path.join(data_path, "dev.tsv"))

    datasets = []
    for raw_data in [raw_train, raw_dev]:
        data = []
        for row in raw_data:
            label = tokenization.convert_to_unicode(row[1])
            text = tokenization.convert_to_unicode(row[3])
            data.append([label, text])
        input_ids, token_type_ids, labels = prep_single_sentence_data(data, vocab_path, max_seq_len)
        datasets.append(GlueDataset(input_ids, token_type_ids, labels))

    train_dataset, dev_dataset = datasets
    eval_metrics = [('accuracy', accuracy)]
    num_classes = 2

    return num_classes, train_dataset, dev_dataset, eval_metrics










