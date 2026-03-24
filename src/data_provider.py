import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class MTSDataset(Dataset):
    def __init__(self, root_path, dataset_name, flag, seq_len, pred_len):
        file_path = os.path.join(root_path, dataset_name, f"{flag}.npy")
        self.data = np.load(file_path).astype(np.float32)
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.T, self.N = self.data.shape
    def __len__(self):
        return self.T - self.seq_len - self.pred_len + 1
    def __getitem__(self, index):
        x = self.data[index : index + self.seq_len]
        y = self.data[index + self.seq_len : index + self.seq_len + self.pred_len]
        return torch.from_numpy(x), torch.from_numpy(y)
    
def data_provider(root_path, dataset_name, flag, seq_len, pred_len, batch_size, shuffle=True):
    dataset = MTSDataset(root_path=root_path, dataset_name=dataset_name,
                         flag=flag, seq_len=seq_len, pred_len=pred_len)
    if flag == "test":
        shuffle_flag = False
    else:
        shuffle_flag = shuffle
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle_flag, drop_last=False, num_workers=4, pin_memory=True)
    return dataset, dataloader