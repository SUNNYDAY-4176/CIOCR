import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

from torchvision import models
from .backbone import Encoder, Decoder, Bottleneck

from .RFBmodule import *

CHANNEL_EXPAND = {
    'resnet18': 1,
    'resnet34': 1,
    'resnet50': 4,
    'resnet101': 4
}


def Soft_aggregation(ps, max_obj):
    num_objects, H, W = ps.shape
    em = torch.zeros(1, max_obj + 1, H, W).to(ps.device)
    em[0, 0, :, :] = torch.prod(1 - ps, dim=0)  # bg prob
    em[0, 1:num_objects + 1, :, :] = ps  # obj prob
    em = torch.clamp(em, 1e-7, 1 - 1e-7)
    logit = torch.log((em / (1 - em)))

    return logit


class ResBlock(nn.Module):
    def __init__(self, indim, outdim=None, stride=1):
        super(ResBlock, self).__init__()
        if outdim == None:
            outdim = indim
        if indim == outdim and stride == 1:
            self.downsample = None
        else:
            self.downsample = nn.Conv2d(indim, outdim, kernel_size=3, padding=1, stride=stride)

        self.conv1 = nn.Conv2d(indim, outdim, kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(outdim, outdim, kernel_size=3, padding=1)

    def forward(self, x):
        r = self.conv1(F.relu(x))
        r = self.conv2(F.relu(r))

        if self.downsample is not None:
            x = self.downsample(x)

        return x + r


class Encoder_M(nn.Module):
    def __init__(self, arch):
        super(Encoder_M, self).__init__()
        self.conv1_m = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.conv1_bg = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        resnet = models.__getattribute__(arch)(pretrained=True)
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool

        self.res2 = resnet.layer1
        self.res3 = resnet.layer2
        self.res4 = resnet.layer3

        self.register_buffer('mean', torch.FloatTensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.FloatTensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, in_f, in_m, in_bg):
        f = in_f
        m = torch.unsqueeze(in_m, dim=1).float()
        bg = torch.unsqueeze(in_bg, dim=1).float()

        x = self.conv1(f) + self.conv1_m(m) + self.conv1_bg(bg)
        x = self.bn1(x)
        c1 = self.relu(x)
        x = self.maxpool(c1)
        r2 = self.res2(x)
        r3 = self.res3(r2)
        r4 = self.res4(r3)

        return r4, r3, r2, c1


class Encoder_Q(nn.Module):
    def __init__(self, arch):
        super(Encoder_Q, self).__init__()

        resnet = models.__getattribute__(arch)(pretrained=True)
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool

        self.res2 = resnet.layer1
        self.res3 = resnet.layer2
        self.res4 = resnet.layer3

        self.register_buffer('mean', torch.FloatTensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.FloatTensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, in_f):
        f = in_f

        x = self.conv1(f)
        x = self.bn1(x)
        c1 = self.relu(x)
        x = self.maxpool(c1)
        r2 = self.res2(x)
        r3 = self.res3(r2)
        r4 = self.res4(r3)

        return r4, r3, r2, c1


class Refine(nn.Module):
    def __init__(self, inplanes, planes):
        super(Refine, self).__init__()
        self.convFS = nn.Conv2d(inplanes, planes, kernel_size=(3, 3), padding=(1, 1), stride=1)
        self.ResFS = ResBlock(planes, planes)
        self.ResMM = ResBlock(planes, planes)

    def forward(self, f, pm):
        s = self.ResFS(self.convFS(f))
        m = s + F.interpolate(pm, size=s.shape[2:], mode='bilinear', align_corners=False)
        m = self.ResMM(m)
        return m


class Decoder(nn.Module):
    def __init__(self, inplane, mdim, expand):
        super(Decoder, self).__init__()
        self.convFM = nn.Conv2d(inplane, mdim, kernel_size=(3, 3), padding=(1, 1), stride=1)
        self.ResMM = ResBlock(mdim, mdim)
        self.RF3 = Refine(128 * expand, mdim)
        self.RF2 = Refine(64 * expand, mdim)

        self.pred2 = nn.Conv2d(mdim, 2, kernel_size=(3, 3), padding=(1, 1), stride=1)

    def forward(self, r4, r3, r2, f):
        m4 = self.ResMM(self.convFM(r4))
        m3 = self.RF3(r3, m4)
        m2 = self.RF2(r2, m3)

        p2 = self.pred2(F.relu(m2))

        p = F.interpolate(p2, size=f.shape[2:], mode='bilinear', align_corners=False)
        return p




class KeyValue(nn.Module):
    def __init__(self, indim, keydim, valdim):
        super(KeyValue, self).__init__()
        self.Key = nn.Conv2d(indim, keydim, kernel_size=3, padding=1, stride=1)
        self.Value = nn.Conv2d(indim, valdim, kernel_size=3, padding=1, stride=1)

    def forward(self, x):
        return self.Key(x), self.Value(x)


class Conv_decouple(nn.Module):

    def __init__(self, inplanes, planes):
        super(Conv_decouple, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        return out


class STAN(nn.Module):

    def __init__(self, opt):
        super(STAN, self).__init__()

        keydim = opt.keydim
        valdim = opt.valdim
        arch = opt.arch
        expand = CHANNEL_EXPAND[arch]
        self.Encoder_M = Encoder_M(arch)
        self.Encoder_Q = Encoder_Q(arch)
        self.keydim = keydim
        self.valdim = valdim
        self.KV_M_r4 = KeyValue(256 * expand, keydim=keydim, valdim=valdim)
        self.KV_Q_r4 = KeyValue(256 * expand, keydim=keydim, valdim=valdim)
        self.KV_m4 = KeyValue(256 * expand, keydim=keydim, valdim=valdim)
        self.rfb_key = BasicRFB(2 * opt.sampled_frames, opt.sampled_frames)
        self.rfb_val = BasicRFB(2 * opt.sampled_frames, opt.sampled_frames)
        self.Memory1 = Temporal_Memory(decay=opt.temporal_decay)
        self.SpatialMemory = Spatial_Memory(decay=opt.spatial_decay)
        self.conv_decouple = Conv_decouple(2048, 1024)
        self.Decoder = Decoder(2 * valdim, 256, expand)

    def load_param(self, weight):

        s = self.state_dict()
        for key, val in weight.items():
            if key in s and s[key].shape == val.shape:
                s[key][...] = val
            elif key not in s:
                print('ignore weight from not found key {}'.format(key))
            else:
                print('ignore weight of mistached shape in key {}'.format(key))

        self.load_state_dict(s)

    def memorize(self, frame, masks, num_objects):
        frame_batch = []
        mask_batch = []
        bg_batch = []
        for o in range(1, num_objects + 1):
            frame_batch.append(frame)
            mask_batch.append(masks[:, o])

        for o in range(1, num_objects + 1):
            bg_batch.append(torch.clamp(1.0 - masks[:, o], min=0.0, max=1.0))

        # make Batch
        frame_batch = torch.cat(frame_batch, dim=0)
        mask_batch = torch.cat(mask_batch, dim=0)
        bg_batch = torch.cat(bg_batch, dim=0)

        r4, _, _, _ = self.Encoder_M(frame_batch, mask_batch, bg_batch)
        _, c, h, w = r4.size()
        k4, v4 = self.KV_M_r4(r4)
        k4, v4 = k4.reshape(k4.size(0), k4.size(1), -1).permute(0, 2, 1), v4.reshape(v4.size(0), v4.size(1),
                                                                                     -1).permute(0, 2, 1)
        return k4, v4, r4


    def forward(self, frame, mask=None, keys=None, values=None, num_objects=None, max_obj=None,
                opt=None, Clip_idx=None, keys_dict=None, vals_dict=None, patch=2):

        if mask is not None:
            return self.memorize(frame, mask, num_objects)
        else:
            return self.segment(frame, keys, values, num_objects, max_obj, opt, Clip_idx, keys_dict, vals_dict, patch)
