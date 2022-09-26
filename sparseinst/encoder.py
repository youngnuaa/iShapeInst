# Copyright (c) Tianheng Cheng and its affiliates. All Rights Reserved

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from fvcore.nn.weight_init import c2_msra_fill, c2_xavier_fill

from detectron2.utils.registry import Registry
from detectron2.layers import Conv2d

SPARSE_INST_ENCODER_REGISTRY = Registry("SPARSE_INST_ENCODER")
SPARSE_INST_ENCODER_REGISTRY.__doc__ = "registry for SparseInst decoder"


class PyramidPoolingModule(nn.Module):

    def __init__(self, in_channels, channels=512, sizes=(1, 2, 3, 6)):
        super().__init__()
        self.stages = []
        self.stages = nn.ModuleList(
            [self._make_stage(in_channels, channels, size) for size in sizes]
        )
        self.bottleneck = Conv2d(
            in_channels + len(sizes) * channels, in_channels, 1)

    def _make_stage(self, features, out_features, size):
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        conv = Conv2d(features, out_features, 1)
        return nn.Sequential(prior, conv)

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        priors = [F.interpolate(input=F.relu_(stage(feats)), size=(
            h, w), mode='bilinear', align_corners=False) for stage in self.stages] + [feats]
        out = F.relu_(self.bottleneck(torch.cat(priors, 1)))
        return out


class L2Norm(nn.Module):
    def __init__(self, n_channels, scale):
        super(L2Norm, self).__init__()
        self.n_channels = n_channels
        self.gamma = scale or None
        self.eps = 1e-10
        self.weight = nn.Parameter(torch.Tensor(self.n_channels))
        self.reset_parameters()

    def reset_parameters(self):
        init.constant_(self.weight, self.gamma)

    def forward(self, x):
        norm = x.pow(2).sum(dim=1, keepdim=True).sqrt()+self.eps
        x = torch.div(x, norm)
        out = self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(x) * x
        return out


class ConvBnReLU(nn.Module):
    def __init__(self, in_chan, out_chan, ks=3, stride=1, padding=1, *args, **kwargs):
        super(ConvBnReLU, self).__init__()
        self.conv = nn.Conv2d(in_chan,
                              out_chan,
                              kernel_size=ks,
                              stride=stride,
                              padding=padding,
                              bias=False)
        self.bn = nn.BatchNorm2d(out_chan, momentum=0.01)
        self.relu = nn.ReLU(inplace=True)
        self.init_weight()

    def forward(self, x):
        x = self.conv(x)
        # print("bn ",x.shape)
        x = self.bn(x)
        x = self.relu(x)
        return x

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)


class ConvBn(nn.Module):
    def __init__(self, in_chan, out_chan, ks=3, stride=1, padding=1, *args, **kwargs):
        super(ConvBn, self).__init__()
        self.conv = nn.Conv2d(in_chan,
                              out_chan,
                              kernel_size=ks,
                              stride=stride,
                              padding=padding,
                              bias=False)
        self.bn = nn.BatchNorm2d(out_chan, momentum=0.01)
        self.init_weight()

    def forward(self, x):
        x = self.conv(x)
        # print("bn ",x.shape)
        x = self.bn(x)
        return x

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)


class Conv1(nn.Module):
    def __init__(self, in_chan, out_chan, ks=1, stride=1, padding=0, *args, **kwargs):
        super(Conv1, self).__init__()
        self.conv = nn.Conv2d(in_chan,
                              out_chan,
                              kernel_size=ks,
                              stride=stride,
                              padding=padding,
                              bias=False)
        self.init_weight()

    def forward(self, x):
        x = self.conv(x)
        return x

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)


class Convresult(nn.Module):
    def __init__(self, in_chan=256, out_chan=1, ks=1, stride=1, padding=0):
        super(Convresult, self).__init__()
        self.conv = nn.Conv2d(in_chan,
                              out_chan,
                              kernel_size=ks,
                              stride=stride,
                              padding=padding
                              )
        self.init_weight()

    def forward(self, x):
        x = self.conv(x)
        return x

    def init_weight(self):
        nn.init.xavier_normal_(self.conv.weight)
        nn.init.constant_(self.conv.bias, -math.log(0.99 / 0.01))


class FamNet(nn.Module):
    def __init__(self, in_chan, out_chan):
        super(FamNet, self).__init__()
        self.conv11 = Conv1(out_chan, out_chan)
        self.conv12 = Conv1(in_chan, out_chan)
        # self.conv13 = Conv1(in_chan, out_chan)
        self.convbn1 = ConvBnReLU(out_chan * 2, out_chan)
        self.convbn2 = ConvBn(out_chan, out_chan)


    def forward(self, large_map, lower_map):
        h, w = large_map.shape[2], large_map.shape[3]

        large_map_f = self.conv11(large_map)
        lower_map_f = self.conv12(lower_map)
        lower_map_up = F.interpolate(lower_map_f, (h, w), mode='bilinear', align_corners=False)
        feature_cat = torch.cat([large_map_f, lower_map_up], dim=1)
        feature_cat_bn = self.convbn1(feature_cat)
        feature_cat_bn1 = self.convbn2(feature_cat_bn)
        out_feature = feature_cat_bn1 + lower_map_up + large_map
        return out_feature


class PSPModule(nn.Module):
    def __init__(self, in_ch=512, out_ch=512, sizes=(1, 2, 3, 6)):
        super().__init__()
        self.stages = []
        self.stages = nn.ModuleList([self._make_stage(in_ch, size) for size in sizes])
        self.bottleneck = nn.Conv2d(in_ch * (len(sizes) + 1), out_ch, kernel_size=1)
        self.relu = nn.ReLU()

    def _make_stage(self, features, size):
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        conv = nn.Conv2d(features, features, kernel_size=1, bias=False)
        return nn.Sequential(prior, conv)

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        priors = [F.interpolate(input=stage(feats), size=(h, w), mode='bilinear', align_corners=False) for stage in
                  self.stages] + [feats]
        bottle = self.bottleneck(torch.cat(priors, 1))
        return self.relu(bottle)


class FamNetOutput(nn.Module):
    def __init__(self, in_chan, mid_chan, n_classes):
        super(FamNetOutput, self).__init__()
        self.conv = ConvBnReLU(in_chan, mid_chan, ks=3, stride=1, padding=1)
        self.conv_out = Convresult(mid_chan, n_classes)
        self.init_weight()

    def forward(self, x):
        x = self.conv(x)
        x = self.conv_out(x)
        return x

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

    def get_params(self):
        wd_params, nowd_params = [], []
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                wd_params.append(module.weight)
                if not module.bias is None:
                    nowd_params.append(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nowd_params += list(module.parameters())
        return wd_params, nowd_params


@SPARSE_INST_ENCODER_REGISTRY.register()
class InstanceContextEncoder(nn.Module):
    """
    Instance Context Encoder
    1. construct feature pyramids from ResNet
    2. enlarge receptive fields (ppm)
    3. multi-scale fusion
    """

    def __init__(self, cfg, input_shape):
        super().__init__()

        self.psp_size = (32, 64, 128, 256, 512)
        self.num_channels = cfg.MODEL.SPARSE_INST.ENCODER.NUM_CHANNELS
        self.in_channels = [512, 256, 128, 256]


        self.psp = PSPModule(self.psp_size[-1], 512, (1, 2, 3, 6))

        self.fam_16 = FamNet(512, 256)
        self.fam_8 = FamNet(256, 128)
        self.fam_4 = FamNet(128, 64)

        self.fam_4_32 = FamNet(512, 64)
        self.fam_4_16 = FamNet(256, 64)
        self.fam_4_8 = FamNet(128, 64)

        """
        fpn_outputs = []

        for in_channel in reversed(self.in_channels):
            output_conv = Conv2d(in_channel, self.num_channels, 3, padding=1)
            c2_xavier_fill(output_conv)
            fpn_outputs.append(output_conv)
        self.fpn_outputs = nn.ModuleList(fpn_outputs)
        """

        self.init_weight()

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)


    """
    def forward_fuse(self, layers):
        layer1 = layers[0]   #1/4, 256
        layer2 = layers[1]   #1/8, 128
        layer3 = layers[2]   #1/16, 256
        layer4 = layers[3]   #1/32, 512

        layer4 = self.fpn_outputs[-1](layer4)

        layer3 = F.interpolate(layer4, scale_factor=2.0, mode='bilinear', align_corners=False) + self.fpn_outputs[-2](layer3)

        layer2 = F.interpolate(layer3, scale_factor=2.0, mode='bilinear', align_corners=False) + self.fpn_outputs[-3](layer2)

        layer1 = self.fpn_outputs[-4](layer1)

        #weight_out = layer2 + F.interpolate(layer1, scale_factor=0.5, mode='bilinear', align_corners=False)

        inst_out = layer1 + F.interpolate(layer2, scale_factor=2.0, mode='bilinear', align_corners=False)

        return inst_out
    """


    def forward(self, layers):

        #"res2", "res3", "res4", "res5"

        feat4 = layers["res2"]
        feat8 = layers["res3"]
        feat16 = layers["res4"]
        feat32 = layers["res5"]

        psp_feat = self.psp(feat32)  #512
        feat_up16 = self.fam_16(feat16, psp_feat)  #256
        feat_up8 = self.fam_8(feat8, feat_up16)    #128
        feat_up4 = self.fam_4(feat4, feat_up8)

        feat_up4_8 = self.fam_4_8(feat_up4, feat_up8)
        feat_up4_16 = self.fam_4_16(feat_up4, feat_up16)
        feat_up4_32 = self.fam_4_32(feat_up4, psp_feat)

        feat_p = torch.cat([feat_up4, feat_up4_8, feat_up4_16, feat_up4_32], dim=1)  #256

        #inst_out = self.forward_fuse([feat_p, feat_up8, feat_up16, psp_feat])

        return feat_p



def build_sparse_inst_encoder(cfg, input_shape):
    name = cfg.MODEL.SPARSE_INST.ENCODER.NAME
    return SPARSE_INST_ENCODER_REGISTRY.get(name)(cfg, input_shape)
