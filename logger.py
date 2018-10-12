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
    def __init__(self, outf, netF, images_train, masks_train, images_test, masks_test, device=torch.device('cpu')):
        MAX_NUM_SAMPLES = 512
        self.images_train = images_train[:MAX_NUM_SAMPLES]
        self.masks_train = masks_train[:MAX_NUM_SAMPLES]
        self.images_test = images_test
        self.masks_test = masks_test

        self.outf = outf
        self.netF = netF
        self.device = device

        self.iters = []
        self.scores_train = []
        self.scores_test = []

        self.threshold = 0.5

    def flush(self, i):
        self.iters.append(i)
        self.eval_netF()
        self.update_scores()
        self.show_masks(i)

    def eval_netF(self):
        self.netF.eval()
        with torch.no_grad():
            self.masks_train_pred = utils.batch_eval(self.netF, self.images_train, device=self.device).cpu()
            self.masks_test_pred = utils.batch_eval(self.netF, self.images_test, device=self.device).cpu()
        self.netF.train()

    def update_scores(self):
        self.scores_train.append(self.get_score(self.masks_train_pred, self.masks_train))
        self.scores_test.append(self.get_score(self.masks_test_pred, self.masks_test))

        self.plot_curves(self.scores_train, 'train', self.scores_test, 'test', 'score')
        np.save('{}/scores_train.npy'.format(self.outf), np.array(self.scores_train))
        np.save('{}/scores_test.npy'.format(self.outf), np.array(self.scores_test))

    def show_masks(self, i):
        idxs_train = np.random.choice(len(self.masks_train), size=8, replace=False)
        idxs_test = np.random.choice(len(self.masks_test), size=8, replace=False)
        vutils.save_image(torch.cat([self.masks_train[idxs_train], self.masks_train_pred[idxs_train]]),
                          '{}/masks_TRAIN_{}.png'.format(self.outf, i))
        vutils.save_image(torch.cat([self.masks_test[idxs_test], self.masks_test_pred[idxs_test]]),
                          '{}/masks_TEST{}.png'.format(self.outf, i))

    def get_score(self, masks_pred, masks):
        masks_pred = masks_pred.detach().cpu().numpy()
        masks = masks.detach().cpu().numpy()
        return utils.get_score(masks_pred, masks, threshold=self.threshold)

    def plot_curves(self, c1, l1, c2, l2, fname):
        fig, ax = plt.subplots()
        ax.plot(self.iters, c1, label=l1, ls=':')
        ax.plot(self.iters, c2, label=l2)
        ax.legend()
        fig.savefig('{}/{}.png'.format(self.outf, fname))
        plt.close(fig)

class LoggerBCE(Logger):
    """ Extend Logger to include BCE losses """
    def __init__(self, outf, netF, images_train, masks_train, images_test, masks_test, bcefunc, device=torch.device('cpu')):
        Logger.__init__(self, outf, netF, images_train, masks_train, images_test, masks_test, device)
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
    def __init__(self, outf, netF, netD, images_train, masks_train, images_test, masks_test, bcefunc, device=torch.device('cpu')):
        LoggerBCE.__init__(self, outf, netF, images_train, masks_train, images_test, masks_test, bcefunc, device)
        self.netD = netD
        self.ipms = []
        self.alphas = []
        self.omegas = []
        self.iters_status = []
        self.ipms_test = []

        self.update_x_real()

    def update_x_real(self):
        self.x_real = torch.cat([self.images_test, self.masks_test], dim=1)

    def dump(self, i, ipm, alpha, omega):  # to track status of GAN training
        self.iters_status.append(i)
        self.ipms.append(ipm)
        self.alphas.append(alpha)
        self.omegas.append(omega)

    def flush(self, i):
        LoggerBCE.flush(self, i)
        self.update_ganstatus()

    def update_ganstatus(self):
        # self.update_ipms_test()

        WIDTH = 101
        fig, axs = plt.subplots(3, 1, sharex=True)
        axs[0].set_ylabel('IPM')
        axs[0].plot(self.iters_status[WIDTH/2:-WIDTH/2], medfilt(self.ipms, WIDTH)[WIDTH/2:-WIDTH/2])
        # axs[0].plot(self.iters, self.ipms_test, 'x')
        axs[0].set_yscale('symlog')

        axs[1].set_ylabel(r'$\alpha$')
        axs[1].plot(self.iters_status[WIDTH/2:-WIDTH/2], medfilt(self.alphas, WIDTH)[WIDTH/2:-WIDTH/2])

        axs[2].set_ylabel(r'$\Omega$')
        axs[2].semilogy(self.iters_status[WIDTH/2:-WIDTH/2], medfilt(self.omegas, WIDTH)[WIDTH/2:-WIDTH/2])

        axs[2].set_xlabel('iteration')
        fig.savefig('{}/status.png'.format(self.outf))
        plt.close(fig)

        np.save('{}/ipms.npy'.format(self.outf), np.array(self.ipms))
        np.save('{}/alphas.npy'.format(self.outf), np.array(self.alphas))
        np.save('{}/omegas.npy'.format(self.outf), np.array(self.omegas))
        # np.save('{}/ipms_test.npy'.format(self.outf), np.array(self.ipms_test))

    def update_ipms_test(self):
        self.update_x_fake()

        self.netD.eval()
        with torch.no_grad():
            y_real = utils.batch_eval(self.netD, self.x_real, device=self.device).cpu()
            y_fake = utils.batch_eval(self.netD, self.x_fake, device=self.device).cpu()
            ipm = (y_real.mean() - y_fake.mean()).item()
        self.netD.train()
        self.ipms_test.append(ipm)

    def update_x_fake(self):
        self.x_fake = torch.cat([self.images_test, self.masks_test_pred], dim=1)

class LoggerFullGAN(LoggerGAN):
    """ Extend LoggerGAN to include netG """
    def __init__(self, outf, netF, netD, netG, nz, images_train, masks_train, images_test, masks_test, bcefunc, device=torch.device('cpu')):
        LoggerGAN.__init__(self, outf, netF, netD, images_train, masks_train, images_test, masks_test, bcefunc, device)
        self.netG = netG
        self.nz = nz

    def show_generated(self, i):
        self.netG.eval()
        self.netF.eval()
        with torch.no_grad():
            images = self.netG(torch.randn(16,self.nz,1,1).to(self.device))
            masks = self.netF(images)
        self.netG.train()
        self.netF.train()
        pairs = torch.cat([images.view(2,8,*images.shape[1:]), masks.view(2,8,*masks.shape[1:])], dim=1).view(-1,*images.shape[1:])
        vutils.save_image(pairs, '{}/generated_images_{}.png'.format(self.outf, i), nrow=8)

    def update_x_fake(self):
        # TODO
        z = torch.randn(128, self.nz, 1, 1).to(self.device)
        self.netG.eval()
        self.netF.eval()
        with torch.no_grad():
            images = self.netG(z)
            masks = self.netF(images)
        self.netG.train()
        self.netF.train()
        self.x_fake = torch.cat([images, masks], dim=1)

    def flush(self, i):
        LoggerGAN.flush(self, i)
        self.show_generated(i)

class LoggerGANImagesOnly(LoggerFullGAN):
    def __init__(self, outf, netD, netG, nz, images_train, images_test, device=torch.device('cpu')):
        LoggerFullGAN.__init__(self, outf, None, netD, netG, nz, images_train, [None], images_test, [None], None, device)

    def flush(self, i):
        self.iters.append(i)
        self.update_ganstatus()
        self.show_generated(i)

    def update_x_fake(self):
        z = torch.randn(128, self.nz, 1, 1).to(self.device)
        self.netG.eval()
        with torch.no_grad():
            x_fake = self.netG(z)
        self.netG.train()
        self.x_fake = x_fake

    def update_x_real(self):
        self.x_real = self.images_test

class LoggerLovasz(Logger):
    def __init__(self, outf, netF, images_train, masks_train, images_test, masks_test, device=torch.device('cpu')):
        Logger.__init__(self, outf, netF, images_train, masks_train, images_test, masks_test, device)

        self.threshold = 0
        self.losses = []
        self.iters_status = []

    def dump(self, i, loss):
        self.iters_status.append(i)
        self.losses.append(loss)

    def plot_losses(self):
        WIDTH = 101
        fig, ax = plt.subplots()
        ax.set_ylabel('lovasz loss')
        ax.plot(self.iters_status[WIDTH/2:-WIDTH/2], medfilt(self.losses, WIDTH)[WIDTH/2:-WIDTH/2])
        # axs[0].set_yscale('symlog')
        fig.savefig('{}/lovaszloss.png'.format(self.outf))
        plt.close(fig)

    def flush(self, i):
        Logger.flush(self, i)
        self.plot_losses()

    def eval_netF(self):
        Logger.eval_netF(self)
        self.masks_train_pred = (self.masks_train_pred > 0).float()
        self.masks_test_pred = (self.masks_test_pred > 0).float()
