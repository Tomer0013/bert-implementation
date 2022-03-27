import numpy as np
import torch
from torch.utils.data import Dataset


class GlueDataset(Dataset):
    """
    Pytorch dataset for the glue fine-tune tasks.

    Args:
        input_ids (np.ndarray): Numpy array of input data after tokenization and processing.
        token_type_ids (np.ndarray): Numpy array of token type ids.
        labels (np.ndarray): Numpy array of labels.
    """
    def __init__(self, input_ids: np.ndarray, token_type_ids: np.ndarray, labels: np.ndarray) -> None:
        super(GlueDataset, self).__init__()
        self.input_ids = torch.from_numpy(input_ids).int()
        self.token_type_ids = torch.from_numpy(token_type_ids).int()
        self.labels = torch.from_numpy(labels).int()

    def __len__(self) -> int:

        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple:
        example = (
            self.input_ids[idx],
            self.token_type_ids[idx],
            self.labels[idx]
        )

        return example
