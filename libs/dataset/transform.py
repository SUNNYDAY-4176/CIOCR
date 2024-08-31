import numpy as np
import torch
import math
import cv2

import random
import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables.segmaps import SegmentationMapsOnImage
from .data import convert_mask, convert_one_hot, MAX_TRAINING_OBJ


class Compose(object):
    """
    Combine several transformation in a serial manner
    """

    def __init__(self, transform=[]):
        self.transforms = transform

    def __call__(self, imgs, annos):

        for m in self.transforms:
            imgs, annos = m(imgs, annos)

        return imgs, annos


class ToFloat(object):
    """
    convert value type to float
    """

    def __init__(self):
        pass

    def __call__(self, imgs, annos):
        for idx, img in enumerate(imgs):
            imgs[idx] = img.astype(dtype=np.float32, copy=True)

        for idx, anno in enumerate(annos):
            annos[idx] = anno.astype(dtype=np.float32, copy=True)

        return imgs, annos

class Rescale(object):

    """
    rescale the size of image and masks
    """

    def __init__(self, target_size):
        assert isinstance(target_size, (int, tuple, list))
        if isinstance(target_size, int):
            self.target_size = (target_size, target_size)
        else:
            self.target_size = target_size

    def __call__(self, imgs, annos):

        h, w = imgs[0].shape[:2]
        new_height, new_width = self.target_size

        factor = min(new_height / h, new_width / w)
        height, width = int(factor * h), int(factor * w)
        pad_l = (new_width - width) // 2
        pad_t = (new_height - height) // 2

        for id, img in enumerate(imgs):
            canvas = np.zeros((new_height, new_width, 3), dtype=np.float32)
            rescaled_img = cv2.resize(img, (width, height))
            canvas[pad_t:pad_t+height, pad_l:pad_l+width, :] = rescaled_img
            imgs[id] = canvas

        for id, anno in enumerate(annos):
            canvas = np.zeros((new_height, new_width, anno.shape[2]), dtype=np.float32)
            rescaled_anno = cv2.resize(anno, (width, height), cv2.INTER_NEAREST)
            canvas[pad_t:pad_t + height, pad_l:pad_l + width, :] = rescaled_anno
            annos[id] = canvas

        return imgs, annos

class Stack(object):

    """
    stack adjacent frames into input tensors
    """

    def __call__(self, imgs, annos):

        num_img = len(imgs)
        num_anno = len(annos)

        h, w, = imgs[0].shape[:2]

        assert num_img == num_anno
        img_stack = np.stack(imgs, axis=0)
        anno_stack = np.stack(annos, axis=0)

        return img_stack, anno_stack

class ToTensor(object):

    """
    convert to torch.Tensor
    """

    def __call__(self, imgs, annos):

        imgs = torch.from_numpy(imgs.copy())
        annos = torch.from_numpy(annos.astype(np.uint8, copy=True)).float()

        imgs = imgs.permute(0, 3, 1, 2).contiguous()
        annos = annos.permute(0, 3, 1, 2).contiguous()

        return imgs, annos

class Normalize(object):

    def __init__(self):
        self.mean = np.array([0.5, 0.5, 0.5]).reshape([1, 1, 3]).astype(np.float32)
        self.std = np.array([0.5, 0.5, 0.5]).reshape([1, 1, 3]).astype(np.float32)

    def __call__(self, imgs, annos):

        for id, img in enumerate(imgs):
            imgs[id] = (img / 255.0 - self.mean) / self.std

        return imgs, annos


class TrainTransform(object):

    def __init__(self, size):
        self.transform = Compose([
            ToFloat(),
            Rescale(size),
            Normalize(),
            Stack(),
            ToTensor(),
        ])

    def __call__(self, imgs, annos):
        return self.transform(imgs, annos)


class TestTransform(object):

    def __init__(self, size):
        self.transform = Compose([
            ToFloat(),
            Rescale(size),
            Normalize(),
            Stack(),
            ToTensor(),
        ])

    def __call__(self, imgs, annos):
        return self.transform(imgs, annos)

