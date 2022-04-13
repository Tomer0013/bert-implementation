import numpy as np
import re
import string

from collections import Counter


def accuracy(preds: list, labels: list) -> float:
    preds = np.array(preds)
    labels = np.array(labels)
    acc = np.mean(preds == labels)

    return acc


def f1_score(preds: list, labels: list) -> float:

    prec = precision(preds, labels)
    rec = recall(preds, labels)
    score = 2 * (prec * rec) / (prec + rec)

    return score


def precision(preds: list, labels: list) -> float:
    preds = np.array(preds)
    labels = np.array(labels)
    prec = np.mean(preds[preds == 1] == labels[preds == 1])

    return prec


def recall(preds: list, labels: list) -> float:
    preds = np.array(preds)
    labels = np.array(labels)
    rec = np.mean(preds[labels == 1] == labels[labels == 1])

    return rec


def spearman_corr(x: list, y: list) -> float:
    x = np.array(x)
    y = np.array(y)
    n = len(x)
    rank_x = np.zeros(n)
    rank_y = np.zeros(n)
    rank_x[np.argsort(x)] = np.arange(n) + 1
    rank_y[np.argsort(y)] = np.arange(n) + 1
    d_2 = (rank_x - rank_y) ** 2
    rho = 1 - (6 * np.sum(d_2)) / (n * (n ** 2 - 1))

    return rho


def squad_compute_metric_for_eval(metric_fn, pred_answers: dict, eval_items: list) -> list:
    metric_vals = []
    id_answers_dict = {}
    for eval_dict in eval_items:
        if eval_dict['qa_id'] not in id_answers_dict:
            if len(eval_dict['qa_all_answers']) == 0:
                id_answers_dict[eval_dict['qa_id']] = [{'text': ""}]
            else:
                id_answers_dict[eval_dict['qa_id']] = eval_dict['qa_all_answers']
    for key in pred_answers.keys():
        a_pred = pred_answers[key][0]
        real_answers = [x['text'] for x in id_answers_dict[key]]
        metric_vals.append(np.max([metric_fn(a_real, a_pred) for a_real in real_answers]))

    return np.mean(metric_vals)


# All methods below this line are from the official SQuAD 2.0 eval script.
# I have added a "squad_" prefix for the EM and F1 functions.
def normalize_answer(s: str) -> str:
    """Convert to lowercase and remove punctuation, articles and extra whitespace."""

    def remove_articles(text: str) -> str:
        regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
        return re.sub(regex, ' ', text)

    def white_space_fix(text: str) -> str:
        return ' '.join(text.split())

    def remove_punc(text: str) -> str:
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text: str) -> str:
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def get_tokens(s: str) -> list:
    if not s:
        return []

    return normalize_answer(s).split()


def squad_compute_em(a_gold: str, a_pred: str) -> int:

    return int(normalize_answer(a_gold) == normalize_answer(a_pred))


def squad_compute_f1(a_gold: str, a_pred: str) -> float:
    gold_toks = get_tokens(a_gold)
    pred_toks = get_tokens(a_pred)
    common = Counter(gold_toks) & Counter(pred_toks)
    num_same = sum(common.values())
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
        return int(gold_toks == pred_toks)
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)

    return f1
