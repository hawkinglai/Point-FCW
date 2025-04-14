"""
ScanObjectNN download: http://103.24.77.34/scanobjectnn/h5_files.zip
"""

import os
import sys
import glob
import h5py
import numpy as np
from torch.utils.data import Dataset
import fpsample

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

def load_scanobjectnn_data(partition, split=1):
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    all_data = []
    all_label = []
    if split == 1:
        h5_name = BASE_DIR + '/data/h5_files/main_split/' + partition + '_objectdataset.h5'
    else:
        h5_name = BASE_DIR + '/data/h5_files/main_split/' + partition + '_objectdataset_augmentedrot_scale75.h5'
    f = h5py.File(h5_name, mode="r")
    data = f['data'][:].astype('float32')
    label = f['label'][:].astype('int64')
    f.close()
    all_data.append(data)
    all_label.append(label)
    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0)
    return all_data, all_label

def index_points(points, idx):
    """
    Input:
        points: input points data, [N, C]
        idx: sample index data, [S, ]
    Return:
        new_points:, indexed points data, [S, C]
        rest_points:, remaining points data, [N-S, C]
    """
    new_points = points[idx, :]
    mask = np.ones(points.shape[0], dtype=bool)
    mask[idx] = False
    rest_points = points[mask]
    
    return new_points, rest_points



class ScanObjectNN(Dataset):
    def __init__(self, num_points, split=1, partition='training'):
        self.data, self.label = load_scanobjectnn_data(partition=partition, split=split)
        self.num_points = num_points
        self.partition = partition

    def __getitem__(self, item):
        pointcloud_index = fpsample.fps_npdu_kdtree_sampling(self.data[item], self.num_points)
        pointcloud, _ = index_points(self.data[item], pointcloud_index)
        # pointcloud = self.data[item][:self.num_points]
        label = self.label[item]
        if self.partition == 'training':
            np.random.shuffle(pointcloud)
        return pointcloud, label

    def __len__(self):
        return self.data.shape[0]


if __name__ == '__main__':
    train = ScanObjectNN(1024)
    import numpy as np
    test = ScanObjectNN(1024, 'test')
    for data, label in train:
        print(data.shape)
        # np.savetxt('data.txt', data)
        # break
