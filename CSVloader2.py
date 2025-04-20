import torch
from torch.utils.data import Dataset
import pandas as pd

class RowWiseCSVLoader2(Dataset):
    def __init__(self, csv_path, csv_path1, seq):
        self.feature_sizes = seq
        self.feature_sizes.pop()
        
        self.data = pd.read_csv(csv_path)
        self.lb = pd.read_csv(csv_path1)

        start_col = 1  # skip ID column
        self.feature_slices = []

        for size in self.feature_sizes:
            end_col = start_col + size
            self.feature_slices.append((start_col, end_col))
            start_col = end_col

        # Add the final slice for "the rest of the columns"
        total_cols = self.data.shape[1]
        if start_col < total_cols:
            self.feature_slices.append((start_col, total_cols))

        self.feature_arrays = [
            self.data.iloc[:, start:end].values.astype('float32')
            for (start, end) in self.feature_slices
        ]

        self.labels = self.lb.iloc[:, -1].values.astype('int64')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        features = [torch.tensor(f[idx]) for f in self.feature_arrays]
        label = torch.tensor(self.labels[idx])
        return (*features, label)