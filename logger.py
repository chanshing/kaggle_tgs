import numpy as np
from scipy.signal import medfilt

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torchvision.utils as vutils

import utils

class Logger(object):
    """ Score and masks """
    def __init__(self, outf, netG, images_test, masks_test, tanh_mode):
        self.netG = netG
        self.images_test = images_test
        self.masks_test = masks_test
        self.outf = outf

        self.scores_train = []
        self.scores_test = []

        self.tanh_mode = tanh_mode

        self.score_cutoff = 0. if tanh_mode else 0.5

        self.vutils_normalize = True if tanh_mode else False
        self.vutils_range = (-1,1) if tanh_mode else None

    def get_score(self, masks_pred, masks):
        masks_pred = masks_pred.detach().cpu().numpy()
        masks = masks.detach().cpu().numpy()
        return utils.get_score(masks_pred, masks, self.score_cutoff)

    def flush(self, i, masks_pred_train, masks_train):

        self.scores_train.append(self.get_score(masks_pred_train, masks_train).item())

        masks_pred_test = self.get_masks(self.images_test)
        self.scores_test.append(self.get_score(masks_pred_test, self.masks_test).item())

        vutils.save_image(torch.cat([self.masks_test[:8], masks_pred_test[:8]]), '{}/masks_{}.png'.format(self.outf, i), normalize=self.vutils_normalize, range=self.vutils_range)
        vutils.save_image(torch.cat([masks_train[:8], masks_pred_train[:8]]), '{}/masks_train_{}.png'.format(self.outf, i), normalize=self.vutils_normalize, range=self.vutils_range)

        self.plot_curves(self.scores_train, 'train', self.scores_test, 'test', 'score')
        np.save('{}/scores_train.npy'.format(self.outf), np.array(self.scores_train))
        np.save('{}/scores_test.npy'.format(self.outf), np.array(self.scores_test))

    def plot_curves(self, c1, l1, c2, l2, fname):
        fig, ax = plt.subplots()
        ax.plot(c1, label=l1)
        ax.plot(c2, label=l2)
        ax.legend()
        fig.savefig('{}/{}.png'.format(self.outf, fname))
        plt.close(fig)

    def get_masks(self, images):
        self.netG.eval()
        with torch.no_grad():
            masks = self.netG(images)
        self.netG.train()
        return masks

class LoggerBCE(Logger):
    def __init__(self, outf, netG, images_test, masks_test, tanh_mode, bcefunc):
        # super(LoggerBCE, self).__init__(self, outf, netG, images_test, masks_test, score_cutoff)
        Logger.__init__(self, outf, netG, images_test, masks_test, tanh_mode)
        self.losses_train = []
        self.losses_test = []
        self.bcefunc = bcefunc

    def flush(self, i, masks_pred_train, masks_train, loss_train):
        # super(LoggerBCE, self).flush(self, i, masks_pred_train, masks_train)
        Logger.flush(self, i, masks_pred_train, masks_train)

        self.losses_train.append(loss_train)
        masks_pred_test = self.get_masks(self.images_test)
        self.losses_test.append(self.bcefunc(masks_pred_test, self.masks_test).item())

        self.plot_curves(self.losses_train, 'train', self.losses_test, 'test', 'loss')
        np.save('{}/losses_train.npy'.format(self.outf), np.array(self.losses_train))
        np.save('{}/losses_test.npy'.format(self.outf), np.array(self.losses_test))

class LoggerGAN(LoggerBCE):
    def __init__(self, outf, netG, images_test, masks_test, tanh_mode, bcefunc):
        # super(LoggerGAN, self).__init__(self, outf, netG, images_test, masks_test, score_cutoff)
        LoggerBCE.__init__(self, outf, netG, images_test, masks_test, tanh_mode, bcefunc)
        self.losses = []
        self.alphas = []
        self.omegas = []

    def dump(self, loss, alpha, omega):
        self.losses.append(loss)
        self.alphas.append(alpha)
        self.omegas.append(omega)

    def flush(self, i, masks_pred_train, masks_train, bceloss):
        # super(LoggerGAN, self).flush(self, i, masks_pred_train, masks_train)
        LoggerBCE.flush(self, i, masks_pred_train, masks_train, bceloss)
        self.plot_losses()
        self.save_losses()

    def save_losses(self):
        np.save('{}/losses.npy'.format(self.outf), np.array(self.losses))
        np.save('{}/alphas.npy'.format(self.outf), np.array(self.alphas))
        np.save('{}/omegas.npy'.format(self.outf), np.array(self.omegas))

    def plot_losses(self):
        fig, axs = plt.subplots(3, 1, sharex=True)
        axs[0].set_ylabel('IPM')
        axs[0].plot(medfilt(self.losses, 101)[50:-50]); axs[0].set_yscale('symlog')
        axs[1].set_ylabel(r'$\alpha$')
        axs[1].plot(medfilt(self.alphas, 101)[50:-50])
        axs[2].set_ylabel(r'$\Omega$')
        axs[2].semilogy(medfilt(self.omegas, 101)[50:-50])
        axs[2].set_xlabel('iteration')
        fig.savefig('{}/loss.png'.format(self.outf))
        plt.close(fig)
