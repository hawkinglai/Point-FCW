import torch.nn as nn
import torch
from pointnet2_ops import pointnet2_utils
from .tda import TDA_toolkits
from gtda.homology import VietorisRipsPersistence
from gtda.point_clouds import ConsecutiveRescaling

class ConsecutiveRepresentation(nn.Module):
    def __init__(self, factor_init=1.0, metric='euclidean'):
        super().__init__()
        self.factor = factor_init
        self.metric = metric

    def forward(self, X):
        if self.metric == 'euclidean':
            Xt = torch.cdist(X, X, p=2)
            Xt[:, range(Xt.shape[1] - 1), range(1, Xt.shape[1])] *= self.factor
        elif self.metric == 'seuclidean':
            # need a better implemtation
            X = X.cpu().detach().numpy()
            Xt = ConsecutiveRescaling(factor=self.factor, metric=self.metric, n_jobs=-1).fit_transform(X)
            Xt = torch.from_numpy(Xt).cuda().float()
        return Xt
    

class PCSamplingProcessor(nn.Module):
    def __init__(self, group_num, kneighbors):
        super().__init__()
        self.group_num = group_num
        self.kneighbors = kneighbors
    
    def forward(self, xyz):
        B, N, C = xyz.shape
        xyz = xyz.cuda()
        
        idx = pointnet2_utils.furthest_point_sample(xyz, self.group_num).long() 
        # idx = self.farthest_point_sample(xyz, self.group_num)
        new_xyz = self.index_points(xyz, idx)

        knn_idx = self.knn_point(self.kneighbors, xyz, new_xyz)
        grouped_xyz = self.index_points(xyz, knn_idx)

        mean_xyz = new_xyz.unsqueeze(dim=-2)
        std_xyz = torch.std(grouped_xyz - mean_xyz)
        
        knn_xyz = (grouped_xyz - mean_xyz) / (std_xyz + 1e-5)
        knn_xyz = torch.cat([knn_xyz, new_xyz.view(B, self.group_num, 1, -1).repeat(1, 1, self.kneighbors, 1)], dim=-1)
        
        return new_xyz, knn_xyz

    def farthest_point_sample(self, xyz, npoint):
        """
        Input:
            xyz: pointcloud data, [B, N, 3]
            npoint: number of samples
        Return:
            centroids: sampled pointcloud index, [B, npoint]
        """
        device = xyz.device
        B, N, C = xyz.shape
        centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
        distance = torch.ones(B, N).to(device) * 1e10
        farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
        batch_indices = torch.arange(B, dtype=torch.long).to(device)
        for i in range(npoint):
            centroids[:, i] = farthest
            centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
            dist = torch.sum((xyz - centroid) ** 2, -1)
            distance = torch.min(distance, dist)
            farthest = torch.max(distance, -1)[1]
        return centroids

    def index_points(self, points, idx):
        """
        index_point with rest point version.
        Input:
            points: input points data, [B, N, C]
            idx: sample index data, [B, S]
        Return:
            new_points:, indexed points data, [B, S, C]
            rest_points:, remaining points data, [B, N-S, C]
        """
        device = points.device
        B = points.shape[0]
        view_shape = list(idx.shape)
        view_shape[1:] = [1] * (len(view_shape) - 1)
        repeat_shape = list(idx.shape)
        repeat_shape[0] = 1
        batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
        new_points = points[batch_indices, idx, :]
        return new_points

    def knn_point(self, nsample, xyz, new_xyz):
        """
        Input:
            nsample: max sample number in local region
            xyz: all points, [B, N, C]
            new_xyz: query points, [B, S, C]
        Return:
            group_idx: grouped points index, [B, S, nsample]
        """
        sqrdists = self.square_distance(new_xyz, xyz)
        _, group_idx = torch.topk(sqrdists, nsample, dim=-1, largest=False, sorted=False)
        return group_idx
    
    def square_distance(self, src, dst):
        """
        Calculate Euclid distance between each two points.
        src^T * dst = xn * xm + yn * ym + zn * zm；
        sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
        sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
        dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
            = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
        Input:
            src: source points, [B, N, C]
            dst: target points, [B, M, C]
        Output:
            dist: per-point square distance, [B, N, M]
        """
        B, N, _ = src.shape
        _, M, _ = dst.shape
        dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
        dist += torch.sum(src ** 2, -1).view(B, N, 1)
        dist += torch.sum(dst ** 2, -1).view(B, 1, M)
        return dist
    
class Pooling(nn.Module):
    def __init__(self, out_dim):
        super().__init__()
        self.out_transform = nn.Sequential(
                nn.BatchNorm1d(out_dim),
                nn.ReLU(inplace=True))

    def forward(self, knn_x_w):
        lc_x = knn_x_w.max(-1)[0] + knn_x_w.mean(-1)
        lc_x = self.out_transform(lc_x)
        return lc_x
    
class Pooling2(nn.Module):
    def __init__(self, out_dim):
        super().__init__()
        self.out_transform = nn.Sequential(
                nn.BatchNorm1d(out_dim),
                nn.ReLU(inplace=True))

    def forward(self, knn_x_w):
        lc_x = knn_x_w.reshape(knn_x_w.shape[0], -1)
        lc_x = self.out_transform(lc_x)
        return lc_x

class PointTDA(nn.Module):
    def __init__(self, input_points, k_neighbors, factor, metric, dataset):
        super().__init__()
        self.pc_num = input_points
        self.k_neighbors = k_neighbors
        self.factor = factor
        self.metric = metric
        self.dataset = dataset

        self.pc_sampling = PCSamplingProcessor(self.pc_num, self.k_neighbors)
        self.tfcw = ConsecutiveRepresentation(self.factor, self.metric)
        self.pooling = Pooling(6)
        self.pooling2 = Pooling2(36)
        self.vrp = VietorisRipsPersistence(
            metric="precomputed",
            homology_dimensions=[0],
            n_jobs=-1,
            collapse_edges=True
        )
        self.tda = TDA_toolkits(self.dataset)

    def feature_bank_generator(self, feature_list):
        feature_list = torch.cat(feature_list, dim=-1)
        feature_list /= feature_list.norm(dim=-1, keepdim=True)
        return feature_list

    def forward(self, xyz):
        xyz, knn_xyz = self.pc_sampling(xyz)
        knn_xyz = knn_xyz.permute(0, 3, 1, 2) 
        knn_xyz = self.pooling(knn_xyz)
        tfcw = self.tfcw(knn_xyz)

        # tfcw tda
        pd_tfcw = self.vrp.fit_transform(tfcw.cpu().detach().numpy())
        tda_tfcw = torch.from_numpy(self.tda.forward(pd_tfcw))
        
        # tfcw
        tfcw_feature = self.pooling2(tfcw).cpu()

        x = self.feature_bank_generator([tda_tfcw, tfcw_feature])

        return x
    

if __name__ == '__main__':
    xyz = torch.rand(2, 512, 3).cuda()
    model = PointTDA(input_points=256, k_neighbors=90, factor=0.2, dataset='scan3', metric='euclidean').cuda()
    print(model(xyz).shape)