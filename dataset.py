import torch
from torch.utils.data import Dataset


class ImageData(Dataset):
    def __init__(self, data_mat, y_mat):
        self.x = list(zip(data_mat, y_mat))

    def __getitem__(self, idx):
        assert idx < len(self.x)
        return self.x[idx]

    def __len__(self):
        return len(self.x)
