import os
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class TrainBlocks(Dataset):
    def __init__(self, path, totorch=True, random_flip=True):
        self.path = Path(path)
        self.length = len(os.listdir(path))
        self.totorch = totorch
        self.random_flip = random_flip

        # print(f'folder has {self.length} items')

    def __getitem__(self, index):
        data = np.load(self.path / f"data{index}.npy", allow_pickle=True).item()
        x = data["x"]
        y = data["y"]

        if self.random_flip:
            # x can also be in form (feature,x,y,z)
            dimoffset = x.ndim == 4
            for dim in range(3):
                if random.choice([True, False]):
                    x = np.flip(x, axis=dimoffset + dim)
                    y = np.flip(y, axis=dim)

        if self.totorch:
            torch_x = torch.from_numpy(x.astype("float32"))
            if x.ndim == 3:
                torch_x = torch_x.view(1, *x.shape)
            torch_y = torch.from_numpy(y.astype("float32")).long()

            return torch_x, torch_y

        return x, y

    def __len__(self):
        return self.length


class TorchList(Dataset):
    def __init__(self, datalist, totorch=True):
        if not isinstance(datalist, list):
            datalist = [datalist]
        self.datalist = datalist
        self.totorch = totorch

    def __getitem__(self, index):
        x = self.datalist[index]
        if self.totorch:
            torch_x = torch.from_numpy(x.astype("float32"))
            if x.ndim == 3:
                torch_x = torch_x.view(1, *x.shape)
            return torch_x
        return x

    def __len__(self):
        return len(self.datalist)
