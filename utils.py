import random
import numpy as np
import scipy.stats as stats
# from scipy.signal import medfilt
from itertools import izip

import skimage.io as io
import pandas
from sklearn.model_selection import train_test_split
# import skimage.transform as ST

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

def load_seismic_data(csv_file, root_dir, test_size=None, random_state=42):
    idxs = pandas.read_csv(csv_file, index_col="id", usecols=[0]).index
    images = np.array([io.imread('{}/images/{}.png'.format(root_dir, i), as_grey=True).astype(np.float32)[None,...] for i in idxs])
    masks = np.array([io.imread('{}/masks/{}.png'.format(root_dir, i), as_grey=True).astype(bool).astype(np.float32)[None,...] for i in idxs])

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

def get_score(masks, masks_pred, cutoff=0.5):

    masks_pred = masks_pred > cutoff
    masks = masks > cutoff

    thresholds = np.arange(0.5, 1, 0.05)
    scores = []

    for m, mp in izip(masks, masks_pred):
        intersection = np.logical_and(m, mp)
        union = np.logical_or(m, mp)
        if np.any(union):
            iou = float(np.sum(intersection)) / (np.sum(union) + 1e-16)
        else:
            iou = 1.0

        scores.append(np.mean(iou > thresholds))

    return np.mean(scores)

class SmoothBinary(object):
    """ Smooth a binary distribution. Must be in [0,1] range """
    def __init__(self, scale=0.1):
        self.scale = 0.1

    def __call__(self, tensor):
        noise = torch.randn_like(tensor).abs() * self.scale
        tensor = (tensor - (2*tensor - 1)*noise).clamp_(0,1)
        return tensor
