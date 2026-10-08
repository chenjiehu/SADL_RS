import math
import random

import cv2
import numpy as np
import torch


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, label):
        for t in self.transforms:
            image, label = t(image, label)
        return image, label


class ToTensor:
    def __call__(self, image, label):
        if image.ndim == 2:
            image = image[:, :, None]
        image = torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))).float()
        label = torch.from_numpy(np.ascontiguousarray(label)).long()
        return image, label


class Normalize:
    def __init__(self, mean, std):
        self.mean = torch.tensor(mean, dtype=torch.float32).view(-1, 1, 1)
        self.std = torch.tensor(std, dtype=torch.float32).view(-1, 1, 1)

    def __call__(self, image, label):
        return (image - self.mean) / self.std, label


class RandScale:
    def __init__(self, scale, aspect_ratio=None):
        self.scale = scale
        self.aspect_ratio = aspect_ratio

    def __call__(self, image, label):
        s = random.uniform(self.scale[0], self.scale[1])
        ratio = 1.0
        if self.aspect_ratio is not None:
            ratio = math.sqrt(random.uniform(self.aspect_ratio[0], self.aspect_ratio[1]))
        fx, fy = s * ratio, s / ratio
        image = cv2.resize(image, None, fx=fx, fy=fy, interpolation=cv2.INTER_LINEAR)
        label = cv2.resize(label, None, fx=fx, fy=fy, interpolation=cv2.INTER_NEAREST)
        return image, label


class Crop:
    def __init__(self, size, crop_type='rand', padding=None, ignore_label=255,
                 keep_class=1, keep_ratio=0.85, max_retries=30):
        self.crop_h, self.crop_w = (size, size) if isinstance(size, int) else tuple(size)
        self.crop_type = crop_type
        self.padding = padding
        self.ignore_label = ignore_label
        self.keep_class = keep_class
        self.keep_ratio = keep_ratio
        self.max_retries = max_retries

    def _offsets(self, h, w):
        if self.crop_type == 'rand':
            return random.randint(0, h - self.crop_h), random.randint(0, w - self.crop_w)
        return (h - self.crop_h) // 2, (w - self.crop_w) // 2

    def __call__(self, image, label):
        h, w = label.shape
        pad_h, pad_w = max(self.crop_h - h, 0), max(self.crop_w - w, 0)
        if pad_h or pad_w:
            top, left = pad_h // 2, pad_w // 2
            image = cv2.copyMakeBorder(image, top, pad_h - top, left, pad_w - left,
                                       cv2.BORDER_CONSTANT, value=self.padding)
            label = cv2.copyMakeBorder(label, top, pad_h - top, left, pad_w - left,
                                       cv2.BORDER_CONSTANT, value=self.ignore_label)
            h, w = label.shape

        target = self.keep_ratio * np.sum(label == self.keep_class)
        for _ in range(self.max_retries + 1):
            y, x = self._offsets(h, w)
            image_crop = image[y:y + self.crop_h, x:x + self.crop_w]
            label_crop = label[y:y + self.crop_h, x:x + self.crop_w]
            if np.sum(label_crop == self.keep_class) >= target:
                break

        if label_crop.shape != (self.crop_h, self.crop_w):
            image_crop = cv2.resize(image_crop, (self.crop_w, self.crop_h), interpolation=cv2.INTER_LINEAR)
            label_crop = cv2.resize(label_crop, (self.crop_w, self.crop_h), interpolation=cv2.INTER_NEAREST)
        return image_crop, label_crop


class RandRotate:
    def __init__(self, rotate, padding, ignore_label=255, p=0.5):
        self.rotate = rotate
        self.padding = padding
        self.ignore_label = ignore_label
        self.p = p

    def __call__(self, image, label):
        if random.random() < self.p:
            angle = random.uniform(self.rotate[0], self.rotate[1])
            h, w = label.shape
            matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
            image = cv2.warpAffine(image, matrix, (w, h), flags=cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_CONSTANT, borderValue=self.padding)
            label = cv2.warpAffine(label, matrix, (w, h), flags=cv2.INTER_NEAREST,
                                   borderMode=cv2.BORDER_CONSTANT, borderValue=self.ignore_label)
        return image, label


class RandomHorizontalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, label):
        if random.random() < self.p:
            image, label = cv2.flip(image, 1), cv2.flip(label, 1)
        return image, label


class RandomVerticalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, label):
        if random.random() < self.p:
            image, label = cv2.flip(image, 0), cv2.flip(label, 0)
        return image, label


class RandomGaussianBlur:
    def __init__(self, radius=5, p=0.5):
        self.radius = radius
        self.p = p

    def __call__(self, image, label):
        if random.random() < self.p:
            image = cv2.GaussianBlur(image, (self.radius, self.radius), 0)
        return image, label


IMAGENET_MEAN = [0.485 * 255, 0.456 * 255, 0.406 * 255]
IMAGENET_STD = [0.229 * 255, 0.224 * 255, 0.225 * 255]


def build_transforms(crop_size: int, ignore_label: int = 255):
    train = Compose([
        RandScale([0.5, 2.0]),
        RandRotate([-10, 10], padding=IMAGENET_MEAN, ignore_label=ignore_label),
        RandomGaussianBlur(),
        RandomHorizontalFlip(),
        Crop(crop_size, crop_type='rand', padding=IMAGENET_MEAN, ignore_label=ignore_label),
        ToTensor(),
        Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    test = Compose([
        ToTensor(),
        Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return train, test
