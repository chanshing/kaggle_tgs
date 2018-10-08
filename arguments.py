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
        self.add_argument('--niterD', type=int, default=5, help='no. updates of D per update of G')
        self.add_argument('--alpha', type=float, default=1.0, help='Lagrange multiplier')
        self.add_argument('--rho', type=float, default=1e-3, help='quadratic weight penalty')
        self.add_argument('--bce_weight', type=float, default=1.)
        self.add_argument('--no_bce', action='store_true')
        self.add_argument('--archF', default='v0')
        self.add_argument('--archD', default='v0')
        self.add_argument('--num_features_D', type=int, default=4)
        self.add_argument('--dropout', type=float, default=0)
        # --- dataset
        self.add_argument('--csv_file', default='all/train.csv')
        self.add_argument('--root_dir', default='all/train/')
        # --- unet arch
        self.add_argument('--num_features_G', type=int, default=4)
        self.add_argument('--num_residuals', type=int, default=2)
        self.add_argument('--gated', action='store_true')
        self.add_argument('--gate_param', type=float, default=0.)
