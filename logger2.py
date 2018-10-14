from collections import defaultdict

import numpy as np
from scipy.signal import medfilt

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torchvision.utils as vutils

import utils

class LoggerBase(object):
    """ Track status, plot and save it """
    def __init__(self, outf):
        self.outf = outf
        self.status = defaultdict(list)
        self.iters_flush = []
        self.iters_status = []

    def dump(self, i, **kws):
        self.iters_status.append(i)
        for k,v in kws.iteritems():
            self.status[k].append(v)

    def flush(self, i):
        self.iters_flush.append(i)

        self.save_dict(self.status)
        self.plot_dict('status', self.status, self.iters_status, width=101)

    def save_dict(self, d):
        for k,v in d.iteritems():
            np.save('{}/{}.npy'.format(self.outf, k), np.array(v))

    def plot_dict(self, fname, d, x, width=1):
        fig = plt.figure()

        for i, (k,v) in enumerate(d.iteritems()):

            if width > 1:
                _x = x[width/2:-width/2]
                _v = medfilt(v, width)[width/2:-width/2]
            else:
                _x, _v = x, v

            ax = fig.add_subplot(len(d),1,i+1)
            ax.plot(_x, _v)
            ax.set_yscale('symlog')
            ax.set_ylabel(k)

        fig.tight_layout()
        fig.savefig('{}/{}.png'.format(self.outf, fname))

class Logger(LoggerBase):
    def __init__(self, outf, net, x_train, y_train, x_valid, y_valid, device=torch.device('cpu')):
        Logger.__init__(self, outf)

        self.net = net
        self.x_train = x_train
        self.y_train = y_train
        self.x_valid = x_valid
        self.y_valid = y_valid
        self.device = device

        self.y_pred_train = None
        self.y_pred_valid = None
        self.score_fn = None
        self.score = defaultdict(list)

    def flush(self, i):
        Logger.flush(i)
        self.eval()
        self._imshow(i)
        self._update_score()

    def eval(self):
        self.net.eval()
        with torch.no_grad():
            self.y_pred_train = utils.batch_eval(self.net, self.x_train, device=self.device).cpu()
            self.y_pred_valid = utils.batch_eval(self.net, self.x_valid, device=self.device).cpu()
        self.net.train()

    def _imshow(self, i):
        idxs_train = np.random.choice(len(self.y_train), size=8, replace=False)
        idxs_valid = np.random.choice(len(self.y_valid), size=8, replace=False)
        vutils.save_image(torch.cat([self.y_train[idxs_train], self.y_train_pred[idxs_train]]),
                          '{}/output_TRAIN_{}.png'.format(self.outf, i))
        vutils.save_image(torch.cat([self.y_valid[idxs_valid], self.y_valid_pred[idxs_valid]]),
                          '{}/output_VALID_{}.png'.format(self.outf, i))

    def _update_score(self):
        if self.score_fn:
            for k,fn in self.score_fn.iteritems():
                score_train = fn(self.y_pred_train, self.y_train)
                score_valid = fn(self.y_pred_valid, self.y_valid)
                self.score['{}_train'.format(k)].append(score_train)
                self.score['{}_valid'.format(k)].append(score_valid)

            self.save_dict(self.score)
            self.plot_dict('score', self.score, self.iters_flush, width=1)

    def set_score_fn(self, **kws):
        self.score_fn = kws

class LoggerGAN(Logger):
    def __init__(self, outf, net, netG, nz, x_train, y_train, x_valid, y_valid, device=torch.device('cpu')):
        Logger.__init__(self, outf, net, x_train, y_train, x_valid, y_valid, device=device)

        self.netG = netG
        self.nz = nz

    def flush(self, i):
        Logger.flush(i)
        self._imshow_generated(i)

    def _imshow_generated(self, i):
        self.netG.eval()
        with torch.no_grad():
            x = self.netG(torch.randn(64,self.nz,1,1).to(self.device))
        self.netG.train()
        vutils.save_image(x, '{}/generated_images_{}.png'.format(self.outf, i), nrow=8)
