import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from model import PointTDA
from ModelNet import ModelNet40
from ScanObjectNN import ScanObjectNN
from utils import classification
from tqdm import tqdm
import argparse


def get_arguments():
    
    parser = argparse.ArgumentParser()
    # parser.add_argument('--dataset', type=str, default='mn40')
    parser.add_argument('--dataset', type=str, default='scan')

    parser.add_argument('--split', type=int, default=1)
    # parser.add_argument('--split', type=int, default=2)
    # parser.add_argument('--split', type=int, default=3)

    parser.add_argument('--bz', type=int, default=64)  # ModelNet 128 # Aug 16 # orginal 16

    parser.add_argument('--points', type=int, default=512)
    parser.add_argument('--k', type=int, default=120)
    parser.add_argument('--factor', type=float, default=0.3)
    parser.add_argument('--metric', type=str, default='seuclidean')

    args = parser.parse_args()
    return args

@torch.no_grad()
def init():
    print('==> Loading args..')
    args = get_arguments()
    print(args)

    print('==> Preparing model..')
    encoder = PointTDA(input_points=args.points, 
                             k_neighbors=args.k, 
                             factor=args.factor,
                             metric=args.metric,
                             dataset=args.dataset+str(args.split)).cuda()
    encoder.eval()

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
    return args, encoder, train_loader, test_loader

@torch.no_grad()
def FeatureProcessor(args, encoder, dataset_loader):
    print('==> Constructing Memory Bank from ' + args.dataset + ' sets..')
    feature_memory, label_memory = [], []
    import time
    start_time = time.time()

    # with torch.no_grad():
    for points, labels in tqdm(dataset_loader):
        points = points.cuda()
        # Pass through the Non-Parametric Encoder
        point_features = encoder(points)
        feature_memory.append(point_features)

        labels = labels.cuda()
        label_memory.append(labels)     
    end_time = time.time()

    # Calculate the elapsed time
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time} seconds")

    feature_memory = torch.cat(feature_memory, dim=0)
    feature_memory /= feature_memory.norm(dim=-1, keepdim=True)

    label_memory = torch.cat(label_memory, dim=0)
    # label_memory = F.one_hot(label_memory).squeeze().float()
    return [feature_memory, label_memory]

if __name__ == '__main__':
    args, encoder, train_loader, test_loader = init()
    encoder.eval()
    training_memory_list = FeatureProcessor(args, encoder, train_loader)
    test_memory_list = FeatureProcessor(args, encoder, test_loader)
    
    classification(training_memory_list[0], 
                       F.one_hot(training_memory_list[1]).squeeze().float(), 
                       test_memory_list[0],
                       test_memory_list[1],
                       plot=True)
    # save_path = './'
    # fmsp = save_path + 'training_set' + args.dataset + str(args.split) + '_pointtda.pt'
    # tfmsp = save_path + 'test_set' + args.dataset + str(args.split) + '_pointtda.pt'
    # torch.save(training_memory_list[0].cpu(), fmsp)
    # torch.save(test_memory_list[0].cpu(), tfmsp)

    

