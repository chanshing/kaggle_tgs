import os
import random
import numpy as np
import scipy.stats as stats
# from scipy.signal import medfilt
# from itertools import izip

from skimage import transform as stf
from skimage import io
# import pandas
import sklearn
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset
# import torchvision.utils as vutils

def seedme(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def sigmoid2tanh(tensor):
    return 2.0 * (tensor - 0.5)

def tanh2sigmoid(tensor):
    return (tensor + 1.0) / 2.0

class SegmentationDataset(Dataset):
    def __init__(self, images, masks, transform=None):
        assert len(images) == len(masks)
        self.images = images
        self.masks = masks
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        mask = self.masks[idx]

        if self.transform:
            image, mask = self.transform((image, mask))

        return image, mask

class DefaultDataset(Dataset):
    def __init__(self, images, transform=None):
        self.images = images
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]

        if self.transform:
            image = self.transform(image)

        return image


class DataIterator(object):
    def __init__(self, dataset, batch_size=8):
        self.dataset = dataset
        self.batch_size = batch_size

    def __iter__(self):
        return self

    def next(self):
        idxs = np.random.choice(len(self.dataset), size=self.batch_size, replace=False)
        return self.dataset[idxs]

class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, pair):
        image, mask = pair
        if random.random() > self.p:
            # image = np.fliplr(image).copy()
            # mask = np.fliplr(mask).copy()
            image = image[:,:,::-1].copy()
            mask = mask[:,:,::-1].copy()
        return (image, mask)

def concatenate_hflips(images, masks, shuffle=True, random_state=42):
    _images = images[:,:,:,::-1]
    _masks = masks[:,:,:,::-1]
    images = np.concatenate([images, _images], axis=0)
    masks = np.concatenate([masks, _masks], axis=0)

    if shuffle:
        images, masks = sklearn.utils.shuffle(images, masks, random_state=random_state)

    return images, masks

def batch_rotate(img, angle=15, mode='reflect'):
    """ (batch_size, h, w) """
    img = np.moveaxis(img, 0, -1)
    stf.rotate(img, angle=angle, mode=mode)
    img = np.moveaxis(img, -1, 0)
    return img

class GaussSmoothMask(object):
    def __init__(self, sigma=0.1, tanh_mode=True):
        self.sigma = sigma
        self.tanh_mode = tanh_mode

    def __call__(self, pair):
        image, mask = pair
        if not self.tanh_mode:
            mask = sigmoid2tanh(mask)
            mask = gauss_smooth_binary(mask, self.sigma)
            mask = tanh2sigmoid(mask)
        else:
            mask = gauss_smooth_binary(mask, self.sigma)
        return (image, mask)

def gauss_smooth_binary(data, sigma):
    """ Smooth binary data with Gaussian noise. Binary values -1 or 1 """
    noise = stats.halfnorm.rvs(size=data.shape, scale=sigma).astype(np.float32)
    noise = torch.from_numpy(noise)
    data = (data - data*noise).clamp_(-1,1)
    return data

class ToTensor(object):
    def __init__(self):
        pass

    def __call__(self, pair):
        image, mask = pair
        image = torch.from_numpy(image)
        mask = torch.from_numpy(mask)
        return (image, mask)

def load_images(root_dir):
    fnames = os.listdir('{}/images'.format(root_dir))
    # load as grey, float32, unsqueeze
    images = np.array([io.imread('{}/images/{}'.format(root_dir, fname), as_gray=True).astype(np.float32)[None,...] for fname in fnames])
    return images

def load_seismic_data(root_dir, test_size=None, random_state=42):
    fnames = os.listdir('{}/images'.format(root_dir))
    # load as grey, float32, unsqueeze
    images = np.array([io.imread('{}/images/{}'.format(root_dir, fname), as_gray=True).astype(np.float32)[None,...] for fname in fnames])
    masks = np.array([io.imread('{}/masks/{}'.format(root_dir, fname), as_gray=True).astype(bool).astype(np.float32)[None,...] for fname in fnames])

    if test_size:
        # -- compute salt coverage for stratified split
        coverages = np.array([get_coverage(mask) for mask in masks])
        labels = np.array([coverage_to_class(cov) for cov in coverages])
        return train_test_split(images, masks, test_size=test_size, random_state=random_state, stratify=labels)
    else:
        return images, masks

def coverage_to_class(val):
    for i in range(0, 11):
        if val * 10 <= i:
            return i

def get_coverage(mask):
    return np.sum(mask)/(mask.shape[0]*mask.shape[1])

def get_score(masks_pred, masks, threshold=0.5):

    masks_pred = masks_pred > threshold
    masks = masks > threshold

    iou_cuts = np.arange(0.5, 1, 0.05)

    n = masks.shape[0]
    intersection = np.logical_and(masks, masks_pred).reshape(n,-1)
    union = np.logical_or(masks, masks_pred).reshape(n,-1)
    ious = (np.sum(intersection, axis=-1) + 1e-8) / (np.sum(union, axis=-1) + 1e-8)

    scores = np.mean(ious.reshape(n,1) > iou_cuts, axis=-1)

    return np.mean(scores)

class SmoothBinary(object):
    """ Smooth a binary distribution. Must be in [0,1] range """
    def __init__(self, scale=0.1):
        self.scale = scale

    def __call__(self, tensor):
        noise = torch.randn_like(tensor).abs() * self.scale
        tensor = (tensor - (2*tensor - 1)*noise).clamp_(0,1)
        return tensor

def dataiterator(dataloader):
    while True:
        for data in dataloader:
            yield data

def augment(
        # rotation_fn=lambda: np.random.randint(0, 360),
        # translation_fn=lambda: (np.random.randint(-20, 20), np.random.randint(-20, 20)),
        # scale_factor_fn=lambda: np.random.uniform(1,1.25),
        # shear_fn=lambda: np.random.randint(-10, 10)
        rotation_fn=lambda: np.random.randint(-10, 10),
        translation_fn=lambda: (np.random.randint(-10, 10), 0),
        scale_factor_fn=lambda: (np.random.uniform(1,1.25), np.random.uniform(1,1.25)),
        shear_fn=lambda: np.random.randint(-10, 10)
):
    def call(pair):
        if np.random.rand() < .5:
            return pair
        else:
            x1, x2 = pair
            rotation = rotation_fn()
            translation = translation_fn()
            scale = scale_factor_fn()
            shear = shear_fn()

            # do not always scale
            if np.random.randn() <.5:
                scale = None

            # use either one or the other
            if np.random.randn() < .5:
                rotation = 0
            else:
                shear = 0

            tform = stf.AffineTransform(scale=scale, rotation=np.deg2rad(rotation), translation=translation, shear=np.deg2rad(shear))

            x1 = stf.warp(x1, tform, order=1, preserve_range=True, mode='symmetric')
            x2 = stf.warp(x2, tform, order=1, preserve_range=True, mode='symmetric')

            x1, x2 = x1.astype(np.float32), x2.astype(np.float32)

            return (x1, x2)

    return call

def batch_eval(net, dataset, batch_size=128, device=torch.device('cpu')):
    # eval by batches to not blow up memory
    results = [net(x.to(device)) for x in batch_looper(dataset, batch_size=batch_size)]
    results = torch.cat(results, dim=0)
    return results

def batch_looper(alist, batch_size=1):
    l = len(alist)
    for ndx in range(0, l, batch_size):
        yield alist[ndx:min(ndx + batch_size, l)]

def BatchEval(object):
    def __init__(self, net, batch_size=128, device=torch.device('cpu')):
        self.net = net
        self.batch_size = batch_size

    def __call__(self, dataset):
        self.net.eval()
        with torch.no_grad():
            results = [self.net(torch.from_numpy(x).to(self.device)) for x in batch_looper(dataset, batch_size=self.batch_size)]
            results = torch.cat(results, dim=0)
        self.net.train()
        return results
