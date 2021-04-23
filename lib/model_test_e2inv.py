import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import math

import e2cnn.nn as enn
from e2cnn.nn import init
from e2cnn import gspaces


def regular_feature_type(gspace: gspaces.GSpace, planes: int, fixparams: bool = True):
    """ build a regular feature map with the specified number of channels"""
    assert gspace.fibergroup.order() > 0
    
    N = gspace.fibergroup.order()
    
    if fixparams:
        planes *= math.sqrt(N)
    
    planes = planes / N
    planes = int(planes)
    
    return enn.FieldType(gspace, [gspace.regular_repr] * planes)

def trivial_feature_type(gspace: gspaces.GSpace, planes: int, fixparams: bool = True):
    """ build a trivial feature map with the specified number of channels"""
    
    if fixparams:
        planes *= math.sqrt(gspace.fibergroup.order())
        
    planes = int(planes)
    return enn.FieldType(gspace, [gspace.trivial_repr] * planes)

FIELD_TYPE = {
    "trivial": trivial_feature_type,
    "regular": regular_feature_type,
}

def conv5x5(in_type: enn.FieldType, out_type: enn.FieldType, stride=1, padding=2,
            dilation=1, bias=True):
    """5x5 convolution with padding"""
    return enn.R2Conv(in_type, out_type, 5,
                      stride=stride,
                      padding=padding,
                      dilation=dilation,
                      bias=bias,
                      sigma=None,
                    #   frequencies_cutoff=lambda r: 3*r,
                      initialize=False,
                      )

def conv3x3(in_type: enn.FieldType, out_type: enn.FieldType, stride=1, padding=1,
            dilation=1, bias=True, frequencies_cutoff=None):
    """3x3 convolution with padding"""
    return enn.R2Conv(in_type, out_type, 3,
                      stride=stride,
                      padding=padding,
                      dilation=dilation,
                      bias=bias,
                      sigma=None,
                      frequencies_cutoff=frequencies_cutoff,
                      initialize=False,
                      )

class DenseFeatureExtractionModuleE2Inv(nn.Module):
    def __init__(self):
        super(DenseFeatureExtractionModuleE2Inv, self).__init__()

        filters = np.array([32,32, 64,64, 128,128,128, 256,256,256, 512,512,512], dtype=np.int32)*2
        
        # number of rotations to consider for rotation invariance
        N = 8
        
        self.gspace = gspaces.Rot2dOnR2(N)
        self.input_type = enn.FieldType(self.gspace, [self.gspace.trivial_repr] * 3)
        ip_op_types = [
            self.input_type,
        ]
        
        self.num_channels = 64

        for filter_ in filters[:10]:
            ip_op_types.append(FIELD_TYPE['regular'](self.gspace, filter_, fixparams=False))

        self.model = enn.SequentialModule(*[
            conv3x3(ip_op_types[0], ip_op_types[1]),
            enn.ReLU(ip_op_types[1], inplace=True),
            conv3x3(ip_op_types[1], ip_op_types[2]),
            enn.ReLU(ip_op_types[2], inplace=True),
            enn.PointwiseMaxPool(ip_op_types[2], 2),

            conv3x3(ip_op_types[2], ip_op_types[3]),
            enn.ReLU(ip_op_types[3], inplace=True),
            conv3x3(ip_op_types[3], ip_op_types[4]),
            enn.ReLU(ip_op_types[4], inplace=True),
            enn.PointwiseMaxPool(ip_op_types[4], 2),

            conv3x3(ip_op_types[4], ip_op_types[5]),
            enn.ReLU(ip_op_types[5], inplace=True),
            conv3x3(ip_op_types[5], ip_op_types[6]),
            enn.ReLU(ip_op_types[6], inplace=True),
            conv3x3(ip_op_types[6], ip_op_types[7]),
            enn.ReLU(ip_op_types[7], inplace=True),

            enn.PointwiseAvgPool(ip_op_types[7], kernel_size=2, stride=1),

            # conv5x5(ip_op_types[7], ip_op_types[8]),
            # enn.ReLU(ip_op_types[8], inplace=True),
            # conv5x5(ip_op_types[8], ip_op_types[9]),
            # enn.ReLU(ip_op_types[9], inplace=True),
            # conv5x5(ip_op_types[9], ip_op_types[10]),
            # enn.ReLU(ip_op_types[10], inplace=True),
            
            conv3x3(ip_op_types[7], ip_op_types[8], dilation=2, padding=2, frequencies_cutoff=1.),
            enn.ReLU(ip_op_types[8], inplace=True),
            conv3x3(ip_op_types[8], ip_op_types[9], dilation=2, padding=2, frequencies_cutoff=1.),
            enn.ReLU(ip_op_types[9], inplace=True),
            conv3x3(ip_op_types[9], ip_op_types[10], dilation=2, padding=2, frequencies_cutoff=1.),
            enn.ReLU(ip_op_types[10], inplace=True),

            enn.GroupPooling(ip_op_types[10])
        ])

    def forward(self, x):
        x = enn.GeometricTensor(x, self.input_type)
        out = self.model(x)
        out = out.tensor
        return out


class D2NetE2Inv(nn.Module):
    def __init__(self, model_file=None):
        super(D2NetE2Inv, self).__init__()

        self.dense_feature_extraction = DenseFeatureExtractionModuleE2Inv()

        self.detection = HardDetectionModule()

        self.localization = HandcraftedLocalizationModule()

        if model_file is not None:
            self.load_state_dict(torch.load(model_file, map_location='cpu')['model'])

    def forward(self, batch):
        _, _, h, w = batch.size()
        dense_features = self.dense_feature_extraction(batch)

        detections = self.detection(dense_features)

        displacements = self.localization(dense_features)

        return {
            'dense_features': dense_features,
            'detections': detections,
            'displacements': displacements
        }


class HardDetectionModule(nn.Module):
    def __init__(self, edge_threshold=5):
        super(HardDetectionModule, self).__init__()

        self.edge_threshold = edge_threshold

        self.dii_filter = torch.tensor(
            [[0, 1., 0], [0, -2., 0], [0, 1., 0]]
        ).view(1, 1, 3, 3)
        self.dij_filter = 0.25 * torch.tensor(
            [[1., 0, -1.], [0, 0., 0], [-1., 0, 1.]]
        ).view(1, 1, 3, 3)
        self.djj_filter = torch.tensor(
            [[0, 0, 0], [1., -2., 1.], [0, 0, 0]]
        ).view(1, 1, 3, 3)

    def forward(self, batch):
        b, c, h, w = batch.size()
        device = batch.device

        depth_wise_max = torch.max(batch, dim=1)[0]
        is_depth_wise_max = (batch == depth_wise_max)
        del depth_wise_max

        local_max = F.max_pool2d(batch, 3, stride=1, padding=1)
        is_local_max = (batch == local_max)
        del local_max

        dii = F.conv2d(
            batch.view(-1, 1, h, w), self.dii_filter.to(device), padding=1
        ).view(b, c, h, w)
        dij = F.conv2d(
            batch.view(-1, 1, h, w), self.dij_filter.to(device), padding=1
        ).view(b, c, h, w)
        djj = F.conv2d(
            batch.view(-1, 1, h, w), self.djj_filter.to(device), padding=1
        ).view(b, c, h, w)

        det = dii * djj - dij * dij
        tr = dii + djj
        del dii, dij, djj

        threshold = (self.edge_threshold + 1) ** 2 / self.edge_threshold
        is_not_edge = torch.min(tr * tr / det <= threshold, det > 0)

        detected = torch.min(
            is_depth_wise_max,
            torch.min(is_local_max, is_not_edge)
        )
        del is_depth_wise_max, is_local_max, is_not_edge

        return detected


class HandcraftedLocalizationModule(nn.Module):
    def __init__(self):
        super(HandcraftedLocalizationModule, self).__init__()

        self.di_filter = torch.tensor(
            [[0, -0.5, 0], [0, 0, 0], [0,  0.5, 0]]
        ).view(1, 1, 3, 3)
        self.dj_filter = torch.tensor(
            [[0, 0, 0], [-0.5, 0, 0.5], [0, 0, 0]]
        ).view(1, 1, 3, 3)

        self.dii_filter = torch.tensor(
            [[0, 1., 0], [0, -2., 0], [0, 1., 0]]
        ).view(1, 1, 3, 3)
        self.dij_filter = 0.25 * torch.tensor(
            [[1., 0, -1.], [0, 0., 0], [-1., 0, 1.]]
        ).view(1, 1, 3, 3)
        self.djj_filter = torch.tensor(
            [[0, 0, 0], [1., -2., 1.], [0, 0, 0]]
        ).view(1, 1, 3, 3)

    def forward(self, batch):
        b, c, h, w = batch.size()
        device = batch.device

        dii = F.conv2d(
            batch.view(-1, 1, h, w), self.dii_filter.to(device), padding=1
        ).view(b, c, h, w)
        dij = F.conv2d(
            batch.view(-1, 1, h, w), self.dij_filter.to(device), padding=1
        ).view(b, c, h, w)
        djj = F.conv2d(
            batch.view(-1, 1, h, w), self.djj_filter.to(device), padding=1
        ).view(b, c, h, w)
        det = dii * djj - dij * dij

        inv_hess_00 = djj / det
        inv_hess_01 = -dij / det
        inv_hess_11 = dii / det
        del dii, dij, djj, det

        di = F.conv2d(
            batch.view(-1, 1, h, w), self.di_filter.to(device), padding=1
        ).view(b, c, h, w)
        dj = F.conv2d(
            batch.view(-1, 1, h, w), self.dj_filter.to(device), padding=1
        ).view(b, c, h, w)

        step_i = -(inv_hess_00 * di + inv_hess_01 * dj)
        step_j = -(inv_hess_01 * di + inv_hess_11 * dj)
        del inv_hess_00, inv_hess_01, inv_hess_11, di, dj

        return torch.stack([step_i, step_j], dim=1)
