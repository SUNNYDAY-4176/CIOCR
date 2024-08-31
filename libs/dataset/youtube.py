import torch
import os
import math
import cv2
import numpy as np

import json
import yaml
import random
import lmdb
import pickle

from PIL import Image
from ..utils.logger import getLogger
from .data import *
from libs.utils.utility import parse_args

opt, _ = parse_args()


class YoutubeVOS(BaseData):

    def __init__(self, train=True, sampled_frames=3,
                 transform=None, max_skip=2, increment=1, samples_per_video=12):
        data_dir = os.path.join(ROOT, 'Youtube-VOS')

        split = 'train' if train else 'valid'
        self.root = data_dir
        self.imgdir = os.path.join(data_dir, split, 'JPEGImages')
        self.annodir = os.path.join(data_dir, split, 'Annotations')

        with open(os.path.join(data_dir, split, 'meta.json'), 'r') as f:
            meta = json.load(f)

        self.info = meta['videos']
        self.samples_per_video = samples_per_video
        self.sampled_frames = sampled_frames
        self.videos = list(self.info.keys())
        self.length = len(self.videos) * samples_per_video
        self.max_obj = 12

        self.transform = transform
        self.train = train
        self.max_skip = max_skip
        self.increment = increment

    def increase_max_skip(self):
        self.max_skip = min(self.max_skip + self.increment, MAX_TRAINING_SKIP)

    def set_max_skip(self, max_skip):
        self.max_skip = max_skip

    def __getitem__(self, idx):

        vid = self.videos[(idx // self.samples_per_video)]
        imgfolder = os.path.join(self.imgdir, vid)
        annofolder = os.path.join(self.annodir, vid)

        frames = [name[:5] for name in os.listdir(imgfolder)]
        frames.sort()

        sample_frame = frames
        sample_mask = [name[:5] for name in os.listdir(annofolder)]
        sample_mask.sort()

        first_ref = sample_mask[0]
        sample_frame = [sample for sample in sample_frame if int(sample) >= int(first_ref)]
        nframes = len(sample_frame)
        frame = [np.array(Image.open(os.path.join(imgfolder, name + '.jpg'))) for name in sample_frame]
        mask = [np.array(Image.open(os.path.join(annofolder, name + '.png'))) for name in sample_mask]
        num_obj = max([int(msk.max()) for msk in mask])
        for msk in mask:
            msk[msk == 255] = 0
        info = {'name': vid}
        info['frame'] = {
            'imgs': sample_frame,
            'masks': sample_mask
        }
        if not self.train:
            assert len(info['frame']['masks']) == len(
                mask), 'unmatched info-mask pair: {:d} vs {:d} at video {}'.format(len(info['frame']), len(mask), vid)

            num_ref_mask = len(mask)
            mask += [mask[0]] * (nframes - num_ref_mask)
        info['frame']['imgs'].sort()
        info['frame']['masks'].sort()
        info['palette'] = Image.open(os.path.join(annofolder, sample_frame[0] + '.png')).getpalette()
        info['size'] = frame[0].shape[:2]
        mask = [convert_mask(msk, self.max_obj) for msk in mask]

        if self.transform is None:
            raise RuntimeError('Lack of proper transformation')

        frame, mask = self.transform(frame, mask)

        return frame, mask, num_obj, info

    def __len__(self):

        return self.length


register_data('VOS', YoutubeVOS)
