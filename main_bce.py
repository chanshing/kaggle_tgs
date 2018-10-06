# import sys
import os

import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim

from torchvision import transforms

import logger
import models
import arguments
import utils

def main(args):
    utils.seedme(args.seed)
    cudnn.benchmark = True
    os.system('mkdir -p {}'.format(args.outf))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    images_train, images_test, masks_train, masks_test = utils.load_seismic_data(args.csv_file, args.root_dir, test_size=.2, random_state=args.seed)
    dataset_train = utils.SegmentationDataset(images_train, masks_train, transform=transforms.Compose([utils.RandomHorizontalFlip(), utils.ToTensor()]))
    # dataloader = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=1)
    dataiter = utils.DataIterator(dataset_train, batch_size=args.batch_size)

    images_test, masks_test = torch.from_numpy(images_test).to(device), torch.from_numpy(masks_test).to(device)

    netG = models.Unet(num_features=args.num_features_G, num_residuals=args.num_residuals, gated=args.gated, gate_param=args.gate_param).to(device)
    optimizerG = optim.Adam(netG.parameters(), lr=args.lr, amsgrad=True)
    loss_func = torch.nn.BCELoss()

    log = logger.LoggerBCE(args.outf, netG, images_train[:512], masks_train[:512], images_test, masks_test, bcefunc=loss_func)

    # for epoch in range(args.nepoch):
    #     for images, masks in dataloader:

    #         optimizerG.zero_grad()
    #         images, masks = images.to(device), masks.to(device)
    #         masks_pred = netG(images)
    #         loss = loss_func(masks_pred, masks)
    #         loss.backward()
    #         optimizerG.step()

    #     print "[epoch {}/{} | loss: {:.3f}".format(epoch+1, args.nepoch, loss.item())
    #     log.flush(epoch, masks_pred, masks, loss.item())

    for i in range(args.niter):
        optimizerG.zero_grad()
        images, masks = next(dataiter)
        images, masks = images.to(device), masks.to(device)
        masks_pred = netG(images)
        loss = loss_func(masks_pred, masks)
        loss.backward()
        optimizerG.step()

        if (i+1) % args.nprint == 0:
            print "[{}/{} | loss: {:.3f}".format(i+1, args.niter, loss.item())
            log.flush(i+1)

            if (i+1) > 20000:
                torch.save(netG.state_dict(), '{}/netG_iter_{}.pth'.format(args.outf, i+1))

if __name__ == '__main__':
    parser = arguments.BaseParser()
    parser.add_argument('--outf', default='tmp/bce')
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
