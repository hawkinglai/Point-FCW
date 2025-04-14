import os
import glob
import h5py
import numpy as np
from torch.utils.data import Dataset
from collections import defaultdict
import random
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

def load_data(partition):
    DATA_DIR = '/home/kitahara/Desktop/PointTDA/pointtda/data/'
    all_data = []
    all_label = []
    for h5_name in glob.glob(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048', 'ply_data_%s*.h5'%partition)):
    
        f = h5py.File(h5_name,'r')
        data = f['data'][:].astype('float32')
        label = f['label'][:].astype('int64')
        f.close()
        all_data.append(data)
        all_label.append(label)

    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0)

    return all_data, all_label


class ModelNet40(Dataset):
    def __init__(self, num_points, partition='train'):
        self.data, self.label = load_data(partition)
        self.num_points = num_points
        self.partition = partition   
        

    def __getitem__(self, item):
        pointcloud = self.data[item][:self.num_points]
        label = self.label[item]
        if self.partition == 'train':
            np.random.shuffle(pointcloud)
        return pointcloud, label

    def __len__(self):
        return self.data.shape[0]


if __name__ == '__main__':
    train = ModelNet40(1024)
    test = ModelNet40(1024, 'test')

    from torch.utils.data import DataLoader

    import matplotlib.pyplot as plt
    train_loader = DataLoader(ModelNet40(partition='train', num_points=1024), num_workers=4,
                              batch_size=32, shuffle=False, drop_last=True)
    for batch_idx, (data, label) in enumerate(train_loader):
        print(f"batch_idx: {batch_idx}  | data shape: {data.shape} | ;lable shape: {label.shape}")
        # data = data[31]
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        # ax.scatter(data[:, 0], data[:, 1], data[:, 2])
        # plt.show()
        # plt.savefig('/home/kitahara/test/Point-TDA/plot/pc_plot.pdf')
        break

    train_set = ModelNet40(partition='train', num_points=1024)
    test_set = ModelNet40(partition='test', num_points=1024)
    print(f"train_set size {train_set.__len__()}")
    print(f"test_set size {test_set.__len__()}")
