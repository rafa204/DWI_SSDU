import torch
import torch.nn as nn
from configs import Config
from TE_unrolled_net.algo.vamp import vamp_unrolling
from torch.nn.functional import pad

from TE_unrolled_net.build_architecture import build_regularizer, build_dc, build_gamma

conf = Config().parse()
    
class UnrolledNet(nn.Module):
    def __init__(self, size):
        super(UnrolledNet, self).__init__()
        self.conf = conf
        self.R = build_regularizer(conf)     # 1. define Regularizer #
        self.dc = build_dc(conf)             # 2. define DC (Default : shared DC). if set conf.DC_unshared, using unshared DC #
        self.gamma = build_gamma(conf)       # 3. define gamma for ADMM and VAMP #
        ns, nq, mb, nc, nx, ny = size

        self.size = size
        if self.conf.qdim:
            self.input_dims = [1, nq, mb * nx, ny]
            self.output_dims = [nq, mb, nx, ny]
        else:
            self.input_dims = [1, mb * nx, ny]
            self.output_dims = [1, mb, nx, ny]

        self.test = False

        if self.conf.model == "Unet_time_emb":
            self.pady = -(-self.input_dims[-1] // 16) * 16 - self.input_dims[-1]  
            self.padx = -(-self.input_dims[-2] // 16) * 16 - self.input_dims[-2]  
        else:
            self.padx = 0
            self.pady = 0

        #Get CAIPI shifts
        self.caipi_shift = self.conf.fov_shifts
    
    def print_model_size(self):
        # Calculate bytes for parameters and buffers
        param_size = sum(p.nelement() * p.element_size() for p in self.parameters())
        buffer_size = sum(b.nelement() * b.element_size() for b in self.buffers())
    
        total_size_bytes = param_size + buffer_size
        total_size_mb = total_size_bytes / (1024 ** 2)
    
        # Calculate total counts
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        print(f"Total Parameters: {total_params:,}")
        print(f"Trainable Parameters: {trainable_params:,}")
        print(f"Total Model Size: {total_size_mb:.2f} MB")

    def format_data(self, x, dir=1):

        if dir == 1:
            x = self.apply_caipi(x, dir=-1) 
            x = x.reshape(-1, *self.input_dims)
            x = torch.cat((x.real, x.imag), dim=1)
            x = pad(x, (0, self.pady, 0, self.padx), "constant", 0)

        elif dir == -1: 
            x = x[..., 0:self.input_dims[-2], 0:self.input_dims[-1]]
            x_real, x_imag = torch.chunk(x, 2, dim=1)
            x = x_real + 1j * x_imag
            x = x.reshape(-1, *self.output_dims)
            x = self.apply_caipi(x, dir=1)

        return x 

    def apply_caipi(self, x, dir=1):
        slices = []
        for i in range(x.shape[-3]):
            shift = dir * int(self.caipi_shift[i] * self.size[-2])
            slice_i = x[..., i, :, :].roll(shift, dims=-2)
            slices.append(slice_i)
            
        return torch.stack(slices, dim=-3)
     
    def forward(self, args):
        return vamp_unrolling(self, args)



