# ------------------------------------------------------------------------------------
# Stripped version of https://github.com/richzhang/PerceptualSimilarity/tree/master/models
# Copyright (c) 2018, Richard Zhang, Phillip Isola, Alexei A. Efros, Eli Shechtman, Oliver Wang. All rights reserved.
# ------------------------------------------------------------------------------------

from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from collections import namedtuple


LIPIPS_PATH = Path(__file__).parent / "vgg16_lpips.pt"


class LPIPS(nn.Module):
    # Learned perceptual metric
    def __init__(self, use_dropout=True):
        super().__init__()
        self.scaling_layer = ScalingLayer()
        self.chns = [64, 128, 256, 512, 512]  # vg16 features
        self.net = vgg16()
        self.lin0 = NetLinLayer(self.chns[0], use_dropout=use_dropout)
        self.lin1 = NetLinLayer(self.chns[1], use_dropout=use_dropout)
        self.lin2 = NetLinLayer(self.chns[2], use_dropout=use_dropout)
        self.lin3 = NetLinLayer(self.chns[3], use_dropout=use_dropout)
        self.lin4 = NetLinLayer(self.chns[4], use_dropout=use_dropout)
        #【不加载预训练模型】
        # self.load_from_pretrained()
        # for param in self.parameters():
        #     param.requires_grad = False

    def load_from_pretrained(self):
        # vgg16 + lpips
        state = torch.load(LIPIPS_PATH, map_location="cpu")
        self.load_state_dict(state)
        print("loaded pretrained VGG16 LPIPS loss from {}".format(LIPIPS_PATH))

    def forward(self, input, target):
        in0_input, in1_input = (self.scaling_layer(input), self.scaling_layer(target))
        outs0, outs1 = self.net(in0_input), self.net(in1_input)
        feats0, feats1, diffs = {}, {}, {}
        lins = [self.lin0, self.lin1, self.lin2, self.lin3, self.lin4]
        for kk in range(len(self.chns)):
            feats0[kk], feats1[kk] = F.normalize(outs0[kk]), F.normalize(outs1[kk])
            diffs[kk] = (feats0[kk] - feats1[kk]) ** 2

        res = [lins[kk].model(diffs[kk]).mean([1, 2, 3]) for kk in range(len(self.chns))]
        val = res[0]
        for l in range(1, len(self.chns)):
            val += res[l]
        return val


class ScalingLayer(nn.Module):
    def __init__(self):
        super(ScalingLayer, self).__init__()
        self.register_buffer('shift', torch.Tensor([-.030, -.088, -.188])[None, :, None, None])
        self.register_buffer('scale', torch.Tensor([.458, .448, .450])[None, :, None, None])

    def forward(self, inp):
        return (inp - self.shift) / self.scale


class NetLinLayer(nn.Module):
    """ A single linear layer which does a 1x1 conv """
    def __init__(self, chn_in, chn_out=1, use_dropout=False):
        super(NetLinLayer, self).__init__()
        layers = [nn.Dropout(), ] if (use_dropout) else []
        layers += [nn.Conv2d(chn_in, chn_out, 1, stride=1, padding=0, bias=False), ]
        self.model = nn.Sequential(*layers)


class vgg16(torch.nn.Module):
    def __init__(self):
        super(vgg16, self).__init__()

        vgg = models.vgg16(pretrained=False)

        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        self.N_slices = 5
        for x in range(4):
            self.slice1.add_module(str(x), vgg.features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg.features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg.features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg.features[x])
        for x in range(23, 30):
            self.slice5.add_module(str(x), vgg.features[x])

    def forward(self, X):
        h = self.slice1(X)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        h = self.slice5(h)
        h_relu5_3 = h
        vgg_outputs = namedtuple("VggOutputs", ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3', 'relu5_3'])
        out = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3, h_relu5_3)
        return out
