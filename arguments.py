import argparse

class BaseParser(argparse.ArgumentParser):
    def __init__(self):
        super(BaseParser, self).__init__()
        self.add_argument('--quick_test', action='store_true')
        self.add_argument('--nprint', type=int, default=500)
        self.add_argument('--seed', type=int, default=42)
        self.add_argument('--batch_size', type=int, default=8)
        self.add_argument('--niter', type=int, default=20000)
        self.add_argument('--nepoch', type=int, default=50)
        self.add_argument('--lr', type=float, default=1e-3, help='learning rate')
        self.add_argument('--lrG', type=float, default=1e-3, help='learning rate')
        self.add_argument('--lrD', type=float, default=1e-3, help='learning rate')
        self.add_argument('--niterD', type=int, default=5, help='num updates of D per update of G')
        self.add_argument('--niterD0', type=int, default=500, help='num updates of D at start')
        self.add_argument('--alpha', type=float, default=1.0, help='Lagrange multiplier')
        self.add_argument('--rho', type=float, default=1e-3, help='quadratic weight penalty')
        self.add_argument('--bce_weight', type=float, default=1.)
        self.add_argument('--no_bce', action='store_true')
        self.add_argument('--archF', default='v0')
        self.add_argument('--archD', default='v0')
        self.add_argument('--archG', default='v0')
        self.add_argument('--num_features_D', type=int, default=8)
        self.add_argument('--dropout', type=float, default=0)
        # --- dataset
        self.add_argument('--csv_file', default='all/train.csv')
        self.add_argument('--root_dir', default='all/train/')
        self.add_argument('--root_dir_unl', default='all/test/')
        # --- unet arch
        self.add_argument('--num_features_F', type=int, default=16)
        self.add_argument('--num_features_U', type=int, default=16)
        self.add_argument('--num_residuals', type=int, default=1)
        self.add_argument('--gated', action='store_true')
        self.add_argument('--gate_param', type=float, default=0.)
        # --- netG
        self.add_argument('--num_features_G', type=int, default=8)
        self.add_argument('--nz', type=int, default=128)

        self.add_argument('--smooth_noise', type=float, default=0.05)

        self.add_argument('--netG', default=None)
        self.add_argument('--netF', default=None)
        self.add_argument('--netD', default=None)

        self.add_argument('--wdecay', type=float, default=0)
        self.add_argument('--wdecay_v', type=float, default=0)
        self.add_argument('--wdecay_s', type=float, default=0)

        self.add_argument('--lovasz_weight', type=float, default=1.)
        self.add_argument('--alphaF', type=float, default=1.)
        self.add_argument('--rhoF', type=float, default=1e-3)
        self.add_argument('--alphaS', type=float, default=1.)
        self.add_argument('--rhoS', type=float, default=1e-3)
