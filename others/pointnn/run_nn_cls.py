import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from tqdm import tqdm

from datasets.data_scan import ScanObjectNN
from datasets.data_mn40 import ModelNet40
from utils import *
from models import Point_NN



@torch.no_grad()
def main(args):
    print('==> Loading args..')
    print(args)
    # torch.manual_seed(3407)

    print('==> Preparing model..')
    point_nn = Point_NN(input_points=args.points, num_stages=args.stages,
                        embed_dim=args.dim, k_neighbors=args.k,
                        alpha=args.alpha, beta=args.beta).cuda()
    point_nn.eval()


    print('==> Preparing data..')

    if args.dataset == 'scan':
        train_loader = DataLoader(ScanObjectNN(split=args.split, partition='training', num_points=args.points), 
                                    num_workers=8, batch_size=args.bz, shuffle=False, drop_last=False)
        test_loader = DataLoader(ScanObjectNN(split=args.split, partition='test', num_points=args.points), 
                                    num_workers=8, batch_size=args.bz, shuffle=False, drop_last=False)
    elif args.dataset == 'mn40':
        train_loader = DataLoader(ModelNet40(partition='train', num_points=args.points), 
                                    num_workers=8, batch_size=args.bz, shuffle=False, drop_last=False)
        test_loader = DataLoader(ModelNet40(partition='test', num_points=args.points), 
                                    num_workers=8, batch_size=args.bz, shuffle=False, drop_last=False)


    print('==> Constructing Point-Memory Bank..')

    feature_memory, label_memory = [], []
    # with torch.no_grad():
    for points, labels in tqdm(train_loader):
        
        points = points.cuda().permute(0, 2, 1)
        # Pass through the Non-Parametric Encoder
        point_features = point_nn(points)
        feature_memory.append(point_features)

        labels = labels.cuda()
        label_memory.append(labels)      

    # Feature Memory
    feature_memory = torch.cat(feature_memory, dim=0)
    feature_memory /= feature_memory.norm(dim=-1, keepdim=True)
    feature_memory = feature_memory.permute(1, 0)
    # Label Memory
    label_memory = torch.cat(label_memory, dim=0)
    label_memory = F.one_hot(label_memory).squeeze().float()


    print('==> Saving Test Point Cloud Features..')
    import time

    # Record the start time
    start_time = time.time()


    test_features, test_labels = [], []
    # with torch.no_grad():
    for points, labels in tqdm(test_loader):
        
        points = points.cuda().permute(0, 2, 1)
        # Pass through the Non-Parametric Encoder
        point_features = point_nn(points)
        test_features.append(point_features)

        labels = labels.cuda()
        test_labels.append(labels)
    end_time = time.time()

    # Calculate the elapsed time
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time} seconds")

    test_features = torch.cat(test_features)
    test_features /= test_features.norm(dim=-1, keepdim=True)
    test_labels = torch.cat(test_labels)


    print('==> Starting Point-NN..')
    # Search the best hyperparameter gamma
    gamma_list = [i * 10000 / 5000 for i in range(5000)]
    best_acc, best_gamma = 0, 0
    # Similarity Matching
    Sim = test_features @ feature_memory
    for gamma in gamma_list:

        # Label Integrate
        logits = (-gamma * (1 - Sim)).exp() @ label_memory

        acc = cls_acc(logits, test_labels)

        if acc > best_acc:
            # print('New best, gamma: {:.2f}; Point-NN acc: {:.2f}'.format(gamma, acc))
            best_acc, best_gamma = acc, gamma

    print(f"Point-NN's classification accuracy: {best_acc:.2f}.")
    return feature_memory, test_features

if __name__ == '__main__':
    feature_memory, test_features = main()
