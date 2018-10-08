import numpy as np
from scipy.signal import medfilt

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torchvision.utils as vutils

import utils

class Logger(object):
    """ Scores and masks """
    def __init__(self, outf, netF, images_train, masks_train, images_test, masks_test):
        self.netF = netF
        self.images_train = images_train
        self.masks_train = masks_train
        self.images_test = images_test
        self.masks_test = masks_test
        self.outf = outf

        self.num_iters = []
        self.scores_train = []
        self.scores_test = []

    def flush(self, i):
        self.num_iters.append(i)
        self.eval_netG()
        self.update_scores()
        self.show_masks(i)

    def eval_netG(self):
        self.netF.eval()
        with torch.no_grad():
            # self.masks_train_pred = self.netF(self.images_train)
            # self.masks_test_pred = self.netF(self.images_test)
            self.masks_train_pred = utils.batch_eval(self.netF, self.images_train)
            self.masks_test_pred = utils.batch_eval(self.netF, self.images_test)
        self.netF.train()

    def show_masks(self, i):
        idxs_train = np.random.choice(len(self.masks_train), size=8, replace=False)
        idxs_test = np.random.choice(len(self.masks_test), size=8, replace=False)
        vutils.save_image(torch.cat([self.masks_train[idxs_train], self.masks_train_pred[idxs_train]]),
                          '{}/masks_TRAIN_{}.png'.format(self.outf, i))
        vutils.save_image(torch.cat([self.masks_test[idxs_test], self.masks_test_pred[idxs_test]]),
                          '{}/masks_TEST{}.png'.format(self.outf, i))

    def update_scores(self):
        self.scores_train.append(self.get_score(self.masks_train_pred, self.masks_train))
        self.scores_test.append(self.get_score(self.masks_test_pred, self.masks_test))

        self.plot_curves(self.scores_train, 'train', self.scores_test, 'test', 'score')
        np.save('{}/scores_train.npy'.format(self.outf), np.array(self.scores_train))
        np.save('{}/scores_test.npy'.format(self.outf), np.array(self.scores_test))

    def get_score(self, masks_pred, masks):
        masks_pred = masks_pred.detach().cpu().numpy()
        masks = masks.detach().cpu().numpy()
        return utils.get_score(masks_pred, masks)

    def plot_curves(self, c1, l1, c2, l2, fname):
        fig, ax = plt.subplots()
        ax.plot(self.num_iters, c1, label=l1, ls=':')
        ax.plot(self.num_iters, c2, label=l2)
        ax.legend()
        fig.savefig('{}/{}.png'.format(self.outf, fname))
        plt.close(fig)

class LoggerBCE(Logger):
    """ Extend Logger to include BCE losses """
    def __init__(self, outf, netF, images_train, masks_train, images_test, masks_test, bcefunc):
        Logger.__init__(self, outf, netF, images_train, masks_train, images_test, masks_test)
        self.bcefunc = bcefunc
        self.bcelosses_train = []
        self.bcelosses_test = []

    def flush(self, i):
        Logger.flush(self, i)
        self.update_bcelosses()

    def update_bcelosses(self):
        self.bcelosses_train.append(self.bcefunc(self.masks_train_pred, self.masks_train).item())
        self.bcelosses_test.append(self.bcefunc(self.masks_test_pred, self.masks_test).item())

        self.plot_curves(self.bcelosses_train, 'train', self.bcelosses_test, 'test', 'bceloss')
        np.save('{}/bcelosses_train.npy'.format(self.outf), np.array(self.bcelosses_train))
        np.save('{}/bcelosses_test.npy'.format(self.outf), np.array(self.bcelosses_test))

class LoggerGAN(LoggerBCE):
    """ Extend LoggerBCE to include GAN status """
    def __init__(self, outf, netF, images_train, masks_train, images_test, masks_test, bcefunc):
        LoggerBCE.__init__(self, outf, netF, images_train, masks_train, images_test, masks_test, bcefunc)
        self.ipms = []
        self.alphas = []
        self.omegas = []

    def dump(self, ipm, alpha, omega):  # to track status of GAN training
        self.ipms.append(ipm)
        self.alphas.append(alpha)
        self.omegas.append(omega)

    def flush(self, i):
        LoggerBCE.flush(self, i)
        self.update_ganstatus()

    def update_ganstatus(self):
        fig, axs = plt.subplots(3, 1, sharex=True)
        axs[0].set_ylabel('IPM')
        axs[0].plot(medfilt(self.ipms, 101)[50:-50]); axs[0].set_yscale('symlog')
        axs[1].set_ylabel(r'$\alpha$')
        axs[1].plot(medfilt(self.alphas, 101)[50:-50])
        axs[2].set_ylabel(r'$\Omega$')
        axs[2].semilogy(medfilt(self.omegas, 101)[50:-50])
        axs[2].set_xlabel('iteration')
        fig.savefig('{}/status.png'.format(self.outf))
        plt.close(fig)

        np.save('{}/ipms.npy'.format(self.outf), np.array(self.ipms))
        np.save('{}/alphas.npy'.format(self.outf), np.array(self.alphas))
        np.save('{}/omegas.npy'.format(self.outf), np.array(self.omegas))
