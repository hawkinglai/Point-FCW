import torch

from gtda.homology import VietorisRipsPersistence
from gtda.diagrams import (PersistenceEntropy , Amplitude, ForgetDimension,
                           ComplexPolynomial, NumberOfPoints, 
                           Silhouette, BettiCurve, Scaler, Filtering,
                           PairwiseDistance, ForgetDimension,
                           PersistenceLandscape)
from gtda.point_clouds import ConsecutiveRescaling
from gtda.curves import StandardFeatures, Derivative
from sklearn.pipeline import make_union
from gtda.pipeline import Pipeline


class TDA_toolkits:
    def __init__(self, dataset='mn401'):
        self.persistence_entropy = PersistenceEntropy(normalize=True)
        self.nop = NumberOfPoints(n_jobs=-1)
        self.cp_pipeline = Pipeline(steps=[
            ('ForgetDimension',ForgetDimension()),
            ('ComplexPolynomial', ComplexPolynomial(n_coefficients=1, n_jobs=-1))
        ])

        self.plsf_pipeline = Pipeline(steps=[
            ('pl', PersistenceLandscape(n_layers=5, n_bins=100, n_jobs=-1)),
            ('sf', StandardFeatures(function='argmax'))
        ])

        self.bcsf_pipeline = Pipeline(steps=[
            ('bc', BettiCurve(n_bins=100, n_jobs=-1)),
            ('de', Derivative(n_jobs=-1)),
            ('sf', StandardFeatures(function='max'))
        ])

        self.amp_metric_list = [
            {"metric": "bottleneck", "metric_params": {}},
            {"metric": "wasserstein", "metric_params": {"p": 1}},
            {"metric": "wasserstein", "metric_params": {"p": 2}},
            {"metric": "betti", "metric_params": {"p": 1, "n_bins": 100}},
            {"metric": "betti", "metric_params": {"p": 2, "n_bins": 100}},
            {"metric": "landscape", "metric_params": {"n_bins":100, "p":1, "n_layers": 2}},
            {"metric": "landscape", "metric_params": {"n_bins":100, "p":1, "n_layers": 5}},
            {"metric": "landscape", "metric_params": {"n_bins":100, "p":2, "n_layers": 2}},
            {"metric": "landscape", "metric_params": {"n_bins":100, "p":2, "n_layers": 5}},
            {"metric": "silhouette", "metric_params": {"n_bins":100, "p":1}},
            # {"metric": "heat", "metric_params": {"p": 1, "sigma": 1.6, "n_bins": 100}},
            # {"metric": "heat", "metric_params": {"p": 2, "sigma": 1.6, "n_bins": 100}},
            # {"metric": "heat", "metric_params": {"p": 1, "sigma": 3.2, "n_bins": 100}},
            # {"metric": "heat", "metric_params": {"p": 2, "sigma": 3.2, "n_bins": 100}},
        ]

        self.amp = make_union(
            *[Amplitude(**metric, n_jobs=-1) for metric in self.amp_metric_list]
        )

        self.amp_pipeline = Pipeline(steps=[
            ('fd', ForgetDimension()),
            ('amp', self.amp)
        ])

        if dataset == 'scan3':
            self.features_union = make_union(
                self.persistence_entropy, # n
                self.nop, # n
                self.cp_pipeline, # 1*2
                # self.shsf_pipeline,
                self.plsf_pipeline, # 1*5
                self.bcsf_pipeline, # n
                # self.heatkernel,
                self.amp_pipeline # n of amp
            )
        elif dataset == 'scan1':
            self.features_union = make_union(
                self.persistence_entropy, # n
                self.nop, # n
                self.cp_pipeline, # 1*2
                # self.shsf_pipeline,
                self.plsf_pipeline, # 1*5
                self.bcsf_pipeline, # n
                # self.heatkernel,
                self.amp_pipeline # n of amp
            )
        elif dataset == 'mn401':
            self.features_union = make_union(
                self.persistence_entropy, # n
                self.nop, # n
                self.cp_pipeline, # 1*2
                # self.shsf_pipeline,
                self.plsf_pipeline, # 1*5
                self.bcsf_pipeline, # n
                # self.heatkernel,
                self.amp_pipeline # n of amp
            )
        else:
            print('dataset wrong!')


    def forward(self, xyz):
        return self.features_union.fit_transform(xyz)


if __name__ == '__main__':
    data = torch.rand(32, 3, 3)
    print("===> testing Point-TDA ...")
    # model = PC_Sampling_Processor(128, 90, data)
    persistence_wrp = VietorisRipsPersistence(
            metric="precomputed",
            homology_dimensions=[0, 1, 2],
            # weight_params = {"n_neighbors": n_neighbors},   # "p": 1, "r":2
            n_jobs=-1,
            collapse_edges=True
        )
    new_xyz = ConsecutiveRescaling(metric='seuclidean', factor=0.2, n_jobs=-1).fit_transform(data)
    wrp = persistence_wrp.fit_transform(new_xyz)
    processor = TDA_toolkits()
    feature = processor.forward(wrp)
    print(feature.shape)
