import numpy as np
from torch.utils.data import Dataset
import torch


class PointCloudDataset_ZeroPadded(Dataset):
    def __init__(self, file_path):
        self.datafile = np.load(file_path)
        self.data = self.datafile["data"]
        self.labels = self.datafile["labels"]

    # returns array / tensor with the number of hits
    def get_n_points(self, data, axis=0):
        n_points_arr = (data[..., axis] != 0.0).sum()
        return n_points_arr

    def get_means_stds(self):  # assumes [batch, points, feats]
        # mean and std considering (removing) zero padding
        mean_list = []
        std_list = []
        for i in range(self.data.shape[2]):
            data = self.data[..., i].flatten()
            data = data[data != 0.0]
            mean_list.append(data.mean())
            std_list.append(data.std())
        return mean_list, std_list

    def __getitem__(self, idx):
        X = self.data[idx]  # 150, 3
        y = self.labels[idx]  # 1

        # nPoints
        # n = self.get_n_points(X, axis=-1).reshape(1,)

        return {
            "X": X,
            "y": y,
            # 'n' : n,
        }

    def __len__(self):
        return len(self.labels)
