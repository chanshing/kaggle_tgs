# import sys
import os
import time

import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim

from torchvision import transforms

import models
import arguments
import logger
import utils

def main(args):
    utils.seedme(args.seed)
    cudnn.benchmark = True
    os.system('mkdir -p {}'.format(args.outf))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print "Using BCE loss: {}".format(not args.no_bce)

    images_train, images_test, masks_train, masks_test = utils.load_seismic_data(args.root_dir, test_size=.2, random_state=args.seed)
    images_train, masks_train = utils.concatenate_hflips(images_train, masks_train, shuffle=True, random_state=args.seed)
    images_test, masks_test = utils.concatenate_hflips(images_test, masks_test, shuffle=True, random_state=args.seed)

    # transform = transforms.Compose([utils.augment(), utils.ToTensor()])
    transform = transforms.Compose([utils.ToTensor()])
    dataset_train = utils.SegmentationDataset(images_train, masks_train, transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=1)
    dataiter = utils.dataiterator(dataloader)

    netF = models.choiceF[args.archF](num_features=args.num_features_G, num_residuals=args.num_residuals, gated=args.gated, gate_param=args.gate_param).to(device)
    netD = models.choiceD[args.archD](num_features=args.num_features_D, dropout=args.dropout).to(device)
    print netF
    print netD
    optimizerF = optim.Adam(netF.parameters(), lr=args.lr, amsgrad=True)
    optimizerD = optim.Adam(netD.parameters(), betas=(0.5, 0.999), lr=args.lr, amsgrad=True)
    alpha = torch.tensor(args.alpha).to(device)
    loss_func = torch.nn.BCELoss()

    smooth_binary = utils.SmoothBinary(scale=0.1)

    images_test, masks_test = torch.from_numpy(images_test).to(device), torch.from_numpy(masks_test).to(device)
    log = logger.LoggerGAN(args.outf, netF, torch.from_numpy(images_train[:512]).to(device), torch.from_numpy(masks_train[:512]).to(device), images_test, masks_test, bcefunc=loss_func)

    start_time = time.time()
    for i in range(args.niter):

        # --- train D
        for _ in range(args.niterD):
            optimizerD.zero_grad()

            images_real, masks_real = next(dataiter)
            images_real, masks_real = images_real.to(device), masks_real.to(device)
            masks_fake = netF(images_real)
            x_fake = torch.cat((images_real, masks_fake), dim=1)

            # images_real, masks_real = next(dataiter)
            # images_real, masks_real = images_real.to(device), masks_real.to(device)
            masks_real = smooth_binary(masks_real)
            x_real = torch.cat((images_real, masks_real), dim=1)

            x_real.requires_grad_()  # to compute gradD_real
            x_fake.requires_grad_()  # to compute gradD_fake

            y_real = netD(x_real)
            y_fake = netD(x_fake)
            lossE = y_real.mean() - y_fake.mean()

            # grad() does not broadcast so we compute for the sum, effect is the same
            gradD_real = torch.autograd.grad(y_real.sum(), x_real, create_graph=True)[0]
            gradD_fake = torch.autograd.grad(y_fake.sum(), x_fake, create_graph=True)[0]
            omega = 0.5*(gradD_real.view(gradD_real.size(0), -1).pow(2).sum(dim=1).mean() +
                         gradD_fake.view(gradD_fake.size(0), -1).pow(2).sum(dim=1).mean())

            loss = -lossE - alpha*(1.0 - omega) + 0.5*args.rho*(1.0 - omega).pow(2)
            loss.backward()
            optimizerD.step()
            alpha -= args.rho*(1.0 - omega.item())

        # --- train G
        optimizerF.zero_grad()
        images_real, masks_real = next(dataiter)
        images_real, masks_real = images_real.to(device), masks_real.to(device)
        masks_fake = netF(images_real)
        x_fake = torch.cat((images_real, masks_fake), dim=1)
        y_fake = netD(x_fake)
        loss = -y_fake.mean()
        bceloss = loss_func(masks_fake, masks_real)
        if not args.no_bce:
            loss = loss + bceloss * args.bce_weight
        loss.backward()
        optimizerF.step()

        log.dump(lossE.item(), alpha.item(), omega.item())

        if (i+1) % args.nprint == 0:
            print 'Time per loop: {} sec/loop'.format((time.time() - start_time)/args.nprint)

            print "[{}/{}] lossE: {:.3f}, bceloss: {:.3f}, alpha: {:.3f}, omega: {:.3f}".format((i+1), args.niter, lossE.item(), bceloss.item(), alpha.item(), omega.item())

            log.flush(i+1)

            if (i+1) > 20000:
                torch.save(netF.state_dict(), '{}/netF_iter_{}.pth'.format(args.outf, i+1))
                torch.save(netD.state_dict(), '{}/netD_iter_{}.pth'.format(args.outf, i+1))

            start_time = time.time()

if __name__ == '__main__':
    parser = arguments.BaseParser()
    parser.add_argument('--outf', default='tmp/gan')
    args = parser.parse_args()

    if args.quick_test:
        print "Running quick test..."
        args.outf = '{}/tmp'.format(args.outf)
        args.niter = 30
        args.nprint = 10
        args.batch_size = 8
        args.num_features_D = 2
        args.num_features_G = 2

    main(args)
