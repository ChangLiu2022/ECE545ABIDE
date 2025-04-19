import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd

class RowWiseCSVLoader(Dataset):
    def __init__(self, csv_path, csv_path1):
        self.data = pd.read_csv(csv_path)
        self.lb = pd.read_csv(csv_path1)
        
        #self.ids = self.data.iloc[:, 0].values  # First column is ID
        self.features = self.data.iloc[:, 1:].values.astype('float32')  # Middle columns are features
        #self.features = self.data.iloc[:, 0:-1].values.astype('float32')  # Middle columns are features
        self.labels = self.lb.iloc[:, -1].values.astype('int64')  # Last column is label

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        #sample_id = self.ids[idx]
        x = torch.tensor(self.features[idx])
        y = torch.tensor(self.labels[idx])
        return x, y