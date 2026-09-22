import argparse
import numpy as np
from pathlib import Path

class Config:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.conf = None

        # directory for the out directory
        self.parser.add_argument('--name', type=str, default='test', help='results file directory')
        self.parser.add_argument('--cuda', type=str, default='3', help='CUDA device to use')
        self.parser.add_argument('--wandb', action="store_true")
        self.parser.add_argument('--wandb_group', type=str, default='DWI_0', help='Name of wandb group to record')

        # hyperparameters for the multi-mask creation and selection 
        self.parser.add_argument('--n_masks', type=int, default=1, help='num masks for multi-mask')
        self.parser.add_argument('--lambda_ratio', type=float, default=0.4, help='sampling ratio for lambda mask') #0.415
        self.parser.add_argument('--lambda_mask', type=float, default=1.0, help='Strength of mask regularization') #0.415
        self.parser.add_argument('--warm_start', action="store_true")
        self.parser.add_argument('--checkpoint', action="store_true")
        self.parser.add_argument('--num_checkpoints', type=int, default=0, help='number of gradient checkpoints')
        self.parser.add_argument('--sense_mask', action="store_true")
        self.parser.add_argument('--mask_batch', action="store_true")
        self.parser.add_argument('--spirit', action="store_true")
        self.parser.add_argument('--ipa', type=int, default=2, help='in plane acceleration rate')

        # Hyperparameters for leaning
        self.parser.add_argument('--model', type=str, default='ResNet_time_emb', help='choose the model {Unet_share, ResNet_share, Unet_multi, ResNet_multi, Unet_time_emb, ResNet_time_emb}')
        self.parser.add_argument('--n_train', type=int, default=640, help='number of slices in training')
        self.parser.add_argument('--n_test', type=int, default=300, help='number of slices in testing')
        self.parser.add_argument('--n_epochs', type=int, default=100, help='number of epochs to train')
        self.parser.add_argument('--lr', type=float, default=3e-4, help='learning rate')
        self.parser.add_argument('--weight_decay', type=float, default=0, help='weight decay rate')
        self.parser.add_argument('--batch_size', type=int, default=1, help='batch size')
        self.parser.add_argument('--qdim', action="store_true")

        #Parameters for validation, plotting, saving models
        self.parser.add_argument('--save_freq', type=int, default=10, help='result saving frequency')
        self.parser.add_argument('--val_freq', type=int, default=10, help='Validation freq')
        self.parser.add_argument('--plot_freq', type=int, default=10, help='Plotting reconstructions freq') 
        self.parser.add_argument('--n_plot', type=int, default=0, help='slices to plot')
        self.parser.add_argument('--seed', type=int, default=0, help='Global random seed')

        # hyperparameters for the unrolled network
        self.parser.add_argument('--nb_unroll_blocks', type=int, default=10, help='number of unrolled blocks')
        self.parser.add_argument('--nb_res_blocks', type=int, default=15, help="number of residual blocks in ResNet")
        self.parser.add_argument('--CG_iter', type=int, default=3, help='number of Conjugate Gradient iterations for DC')

        # ==== Time embedded network parameters
        # For Unrolling Algorithm
        self.parser.add_argument('--Unroll_algo', type=str, default='VAMP', help='Unrolling Algorithm {VSQP, PGD, ADMM, VAMP}')
        self.parser.add_argument('--mu_init', type=float, default=1.5e-2, help='mu init value') #1.5e-2
        self.parser.add_argument('--DC_unshared', action="store_true")
        self.parser.add_argument('--gamma_unshared', action="store_true", help='used in ADMM and VAMP')
        self.parser.add_argument('--gamma_init', type=float, default=1e-1, help='gamma init value used in ADMM and VAMP')
        
        # for Unet
        self.parser.add_argument("--use_norm", type=bool, default=True, help="Use normalization (default: True)")
        self.parser.add_argument('--use_time_emb', action="store_true")
        self.parser.add_argument('--channel_mult', type=int, nargs='+', default=[1, 2, 3], help='number of multi-channels')
        self.parser.add_argument('--num_channels', type=int, default=32, help='number of basic channel')
        
        # for ResNet time-embedding
        self.parser.add_argument('--tau_init', type=float, default=1e-1, help='tau in resnet-te')
        self.parser.add_argument('--time_emb_ratio', type=int, default=2, help='number of basic channel')

    def parse(self, args=None):
        """Parse the configuration"""
        self.conf = self.parser.parse_args(args=args)
        
        fov_num = 3
        self.conf.mb = 5
        self.conf.fov_shifts = np.arange(0, self.conf.mb) / fov_num
#----
        if self.conf.Unroll_algo == "PGD":
            self.conf.mu_init = 2e-1
        elif self.conf.Unroll_algo == "VSQP":
            self.conf.mu_init = 5e-2
        elif self.conf.Unroll_algo == "ADMM":
            self.conf.DC_unshared = True
        elif self.conf.Unroll_algo == "VAMP":
            self.conf.DC_unshared = True
            self.conf.gamma_unshared = True
            if '_share' in self.conf.model or '_multi' in self.conf.model:
                raise ValueError("conf.model should have 'time_emb' instead of 'share' or 'multi' for VAMP")
        
        if "time_emb" in self.conf.model:
            self.conf.use_time_emb = True
#----
        return self.conf
