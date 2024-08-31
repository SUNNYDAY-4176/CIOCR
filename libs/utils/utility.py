import numpy as np
import torch
import os
import os.path as osp
import cv2
import argparse
from PIL import Image
from ..dataset.data import ROOT
from ..config import getCfg, sanity_check
from .logger import getLogger

logger = getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser('Training Mask Segmentation')
    parser.add_argument('--cfg', default='', type=str, help='path to config file')
    parser.add_argument('--local_rank', default=-1, type=int,
                        help='process local rank, only used for distributed training')
    parser.add_argument('options', help='other configurable options', nargs=argparse.REMAINDER)

    args = parser.parse_args()
    opt = getCfg()

    if osp.exists(args.cfg):
        opt.merge_from_file(args.cfg)

    if len(args.options) > 0:
        assert len(args.options) % 2 == 0, 'configurable options must be key-val pairs'
        opt.merge_from_list(args.options)

    sanity_check(opt)

    return opt, args.local_rank


def save_checkpoint(state, epoch, checkpoint='checkpoint', filename='checkpoint'):
    filepath = os.path.join(checkpoint, filename + str(epoch) + '.pth.tar')
    torch.save(state, filepath)
    logger.info('save model at {}'.format(filepath))


def write_mask(mask, info, opt, directory='results', model_name=''):
    name = info['name']

    directory = os.path.join(ROOT, directory)

    if not os.path.exists(directory):
        os.mkdir(directory)

    directory = os.path.join(directory, opt.valset)

    if not os.path.exists(directory):
        os.mkdir(directory)

    directory = os.path.join(directory, model_name)

    if not os.path.exists(directory):
        os.mkdir(directory)

    video = os.path.join(directory, name)
    if not os.path.exists(video):
        os.mkdir(video)

    h, w = info['size']
    th, tw = mask.shape[2:]
    factor = min(th / h, tw / w)
    sh, sw = int(factor * h), int(factor * w)

    pad_l = (tw - sw) // 2
    pad_t = (th - sh) // 2

    for t in range(mask.shape[0]):
        m = mask[t, :, pad_t:pad_t + sh, pad_l:pad_l + sw]
        m = m.transpose((1, 2, 0))
        rescale_mask = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)

        if 'frame' not in info:
            min_t = 0
            step = 1
            output_name = '{:0>5d}.png'.format(t * step + min_t)
        else:
            output_name = '{}.png'.format(info['frame']['imgs'][t])

        if opt.save_indexed_format == 'index':

            rescale_mask = rescale_mask.argmax(axis=2).astype(np.uint8)
            im = Image.fromarray(rescale_mask).convert('P')
            im.putpalette(info['palette'])
            im.save(os.path.join(video, output_name), format='PNG')

        elif opt.save_indexed_format == 'segmentation':

            rescale_mask = rescale_mask.argmax(axis=2).astype(np.uint8)
            seg = np.zeros((h, w, 3), dtype=np.uint8)
            for k in range(1, rescale_mask.max() + 1):
                seg[rescale_mask == k, :] = info['palette'][(k * 3):(k + 1) * 3][::-1]

            inp_img = cv2.imread(
                os.path.join(ROOT, opt.valset, 'JPEGImages', '480p', name, output_name.replace('png', 'jpg')))
            im = cv2.addWeighted(inp_img, 0.5, seg, 0.5, 0.0)
            cv2.imwrite(os.path.join(video, output_name), im)

        elif opt.save_indexed_format == 'heatmap':

            rescale_mask[rescale_mask < 0] = 0.0
            rescale_mask = np.max(rescale_mask[:, :, 1:], axis=2)
            rescale_mask = (rescale_mask - rescale_mask.min()) / (rescale_mask.max() - rescale_mask.min()) * 255
            seg = rescale_mask.astype(np.uint8)
            # seg = cv2.GaussianBlur(seg, ksize=(5, 5), sigmaX=2.5)

            seg = cv2.applyColorMap(seg, cv2.COLORMAP_JET)
            inp_img = cv2.imread(
                os.path.join(ROOT, opt.valset, 'JPEGImages', '480p', name, output_name.replace('png', 'jpg')))
            im = cv2.addWeighted(inp_img, 0.5, seg, 0.5, 0.0)
            cv2.imwrite(os.path.join(video, output_name), im)

        elif opt.save_indexed_format == 'mask':

            fg = np.argmax(rescale_mask, axis=2).astype(np.uint8)

            seg = np.zeros((h, w, 3), dtype=np.uint8)
            seg[fg == 1, :] = info['palette'][3:6][::-1]

            inp_img = cv2.imread(
                os.path.join(ROOT, opt.valset, 'JPEGImages', '480p', name, output_name.replace('png', 'jpg')))
            im = cv2.addWeighted(inp_img, 0.5, seg, 0.5, 0.0)
            cv2.imwrite(os.path.join(video, output_name), im)

        else:
            raise TypeError('unknown save format {}'.format(opt.save_indexed_format))

