import numpy as np
import torch
from torch.utils.data import Dataset


class GlueDataset(Dataset):
    """
    Pytorch dataset for the glue tasks.

    Args:
        input_ids (np.ndarray): Numpy array of input data after tokenization and processing.
        token_type_ids (np.ndarray): Numpy array of token type ids.
        labels (np.ndarray): Numpy array of labels.
        is_regression (bool): Whether the dataset is for classification or regression, and labels
                              will be assigned with the proper data type.
    """
    def __init__(self, input_ids: np.ndarray, token_type_ids: np.ndarray,
                 labels: np.ndarray, is_regression: bool = False) -> None:
        super(GlueDataset, self).__init__()
        self.input_ids = torch.from_numpy(input_ids).long()
        self.token_type_ids = torch.from_numpy(token_type_ids).long()
        if is_regression:
            self.labels = torch.from_numpy(labels).float()
        else:
            self.labels = torch.from_numpy(labels).long()

    def __len__(self) -> int:

        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, ...]:
        example = (
            self.input_ids[idx],
            self.token_type_ids[idx],
            self.labels[idx]
        )

        return example


class SQuADDataset(Dataset):
    """
    Dataset for the SQuAD task.

    Args:
        input_ids (np.ndarray): Numpy array of input data after tokenization and processing.
        token_type_ids (np.ndarray): Numpy array of token type ids.
        start_labels (np.ndarray): Positions of beginning of the answer's span.
        end_labels (np.ndarray): Positions of the end of the answer's span.
    """
    def __init__(self, input_ids: np.ndarray, token_type_ids: np.ndarray,
                 start_labels: np.ndarray, end_labels: np.ndarray) -> None:
        super(SQuADDataset, self).__init__()
        self.input_ids = torch.from_numpy(input_ids).long()
        self.token_type_ids = torch.from_numpy(token_type_ids).long()
        self.start_labels = torch.from_numpy(start_labels).long()
        self.end_labels = torch.from_numpy(end_labels).long()

    def __len__(self) -> int:

        return len(self.start_labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, ...]:
        example = (
            self.input_ids[idx],
            self.token_type_ids[idx],
            self.start_labels[idx],
            self.end_labels[idx]
        )

        return example
