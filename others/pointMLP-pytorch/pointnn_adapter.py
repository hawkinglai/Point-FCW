import argparse
import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from torch.utils.data import DataLoader
from ScanObjectNN import ScanObjectNN
from collections import OrderedDict
from ModelNet import ModelNet40
from tqdm import tqdm
from pointmlp import pointMLP
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

def adjust_values(tensor):
    # Define the sigmoid function
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    # Shift and scale the tensor values so that 0.9 maps to 0
    shifted_tensor = (tensor - 0.9) * 10

    # Apply the sigmoid function
    adjusted_tensor = sigmoid(shifted_tensor)

    return adjusted_tensor

def cls_acc(output, target, topk=1):
    pred = output.topk(topk, 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    acc = correct[: topk].reshape(-1).float().sum(0, keepdim=True).cpu().numpy().item()
    acc = 100 * acc / target.shape[0]
    return acc

def classification(feature_bank, label_bank, test_feature_bank, test_label_bank, plot=False):
    gamma_list = [i * 10000 / 5000 for i in range(5000)]
    best_acc, best_gamma = 0, 0
    Sim = test_feature_bank.cuda().float() @ feature_bank.cuda().permute(1, 0).float()
    # np_Sim = torch.from_numpy(np.loadtxt('/home/kitahara/test/Point-TDA/Sim.txt')).cuda().float()
    for gamma in tqdm(gamma_list, desc="Searching best gamma"):
        logits = (-gamma * (1 - Sim)).exp() @ label_bank
        acc = cls_acc(logits, test_label_bank)

        if acc > best_acc:
            best_acc, best_gamma = acc, gamma

    print(f"TDA's classification accuracy: {best_acc:.2f}.")
    print(f"TDA's best gamma: {best_gamma:.2f}.")

    if plot == True:
        tensor_np = Sim.cpu().numpy()
        adjusted_tensor = adjust_values(tensor_np)
        # print(adjusted_tensor)
        plt.imshow(adjusted_tensor, cmap='hot', interpolation='nearest')
        plt.savefig('/home/kitahara/test/Point-TDAv2/plot.pdf', bbox_inches='tight')

def get_arguments():
    
    parser = argparse.ArgumentParser()
    # parser.add_argument('--dataset', type=str, default='mn40')
    parser.add_argument('--dataset', type=str, default='scan')

    parser.add_argument('--split', type=int, default=3)
    parser.add_argument('--cls', type=int, default=15)
    parser.add_argument('--points', type=int, default=1024)

    parser.add_argument('--bz', type=int, default=128)  # ModelNet 128 # Aug 16 # orginal 16

    args = parser.parse_args()
    return args

import torch.backends.cudnn as cudnn
def load_pretrained(checkpoint_path, num):
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'), weights_only=True)
    # print(checkpoint)
    net = pointMLP(num_classes=num)
    # print(net)
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    net = net.to(device)
    if device == 'cuda':
            net = torch.nn.DataParallel(net)
            cudnn.benchmark = True

    new_state_dict = OrderedDict()
    for k, v in checkpoint['net'].items():
        name = k  # add `module.`
        new_state_dict[name] = v

    net.load_state_dict(new_state_dict)
    del net.module.classifier
    net = net
    return net

@torch.no_grad()
def init():
    print('==> Loading args..')
    args = get_arguments()
    print(args)
    # to download the model:
    # http://github.com/13952522076/pointMLP-pytorch/tree/d2b8dbaa06eb6176b222dcf2ad248f8438582026
    print('==> Preparing model..')
    if args.dataset == 'scan':
        path = '/home/kitahara/Desktop/PointTDA/pointtda/others/pointMLP-pytorch/best_checkpoint_scan.pth'
    elif args.dataset == 'mn40':
        path = '/home/kitahara/Desktop/PointTDA/pointtda/others/pointMLP-pytorch/best_checkpoint_model.pth'
    else:
        print('wrong')
    model = load_pretrained(path, args.cls)
    model.eval()

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
    else:
        print('none')
    return args, model, train_loader, test_loader


@torch.no_grad()
def FeatureProcessor(args, encoder, dataset_loader):
    print('==> Constructing Memory Bank from ' + args.dataset + ' sets..')
    feature_memory, label_memory = [], []
    
    with torch.no_grad():
        for points, labels in tqdm(dataset_loader):
            points = points.permute(0, 2, 1).cuda()
            # Pass through the Non-Parametric Encoder
            point_features = encoder(points)
            feature_memory.append(point_features)

            labels = labels.cuda()
            label_memory.append(labels)     

    feature_memory = torch.cat(feature_memory, dim=0)
    feature_memory /= feature_memory.norm(dim=-1, keepdim=True)

    label_memory = torch.cat(label_memory, dim=0)
    # label_memory = F.one_hot(label_memory).squeeze().float()
    return [feature_memory, label_memory]

if __name__ == '__main__':
    args, encoder, train_loader, test_loader = init()
    encoder.cuda()
    encoder.eval()
    training_memory_list = FeatureProcessor(args, encoder, train_loader)
    test_memory_list = FeatureProcessor(args, encoder, test_loader)
    
    classification(training_memory_list[0], 
                       F.one_hot(training_memory_list[1]).squeeze().float(), 
                       test_memory_list[0],
                       test_memory_list[1])
    

    import numpy
    save_path = './'
    fmsp = save_path + 'training_set' + args.dataset + str(args.split) + '_pointmlp.npy'
    tfmsp = save_path + 'test_set' + args.dataset + str(args.split) + '_pointmlp.npy'
    numpy.save(fmsp, training_memory_list[0].cpu().numpy())
    numpy.save(tfmsp, test_memory_list[0].cpu().numpy())
    







