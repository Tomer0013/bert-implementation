"""
Obviously sklearn metrics could have been used here, but since we're implementing everything from scratch,
might as well go all the way :)
"""

import numpy as np


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
