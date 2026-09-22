import torch
from torch.utils.checkpoint import checkpoint
import numpy as np
import sys
import random

def checkpoint_list(n, m):
    #Create a list of evenly spaces "1"s, which correspond to which DC blocks and Regulatizers we want to checkpoint
    #For example: if we have 6 blocks and want three checkpoints, the list should be [1 0 1 0 1 0] (Approximately)
    if m == 0:
        return [0 for i in range(n)]
    r = n % m
    k = (n - r) // m + 1
    checkpoint_list = [int((i + k//2) % k == 0) for i in range(n)]
    checkpoint_list = np.array(checkpoint_list)
    num_cps = m - np.sum(checkpoint_list)

    for i in range(num_cps):
        checkpoint_list[i*k] = 1

    return checkpoint_list

def vamp_unrolling(self, args):
    CNN_outputs = []
    CNN_inputs = []
    ksp, coil_maps, mask = args[:3]

    zerofilled = self.dc[0].EH(ksp, coil_maps, mask)
    u = zerofilled.clone()
    x0 = None
        
    gamma = []

    # Total sub-blocks: (nb_unroll_blocks + 1) DC steps + nb_unroll_blocks CNN steps
    total_sub_blocks = 2 * self.conf.nb_unroll_blocks + 1
    cp = checkpoint_list(total_sub_blocks, self.conf.num_checkpoints)

    def dc_block(u, x0):
        #Data fidelity block
        if not self.conf.warm_start:
            x0 = None

        output, mu = self.dc[i](zerofilled, coil_maps, mask, denoiser=u, x0 = x0, CG_iter = self.conf.CG_iter)

        if self.conf.warm_start:
            x0 = output.clone()

        return output, mu, x0

    def cnn_block(output, u, i):
        #Regularizer (CNN) block

        # --- gamma selection ---
        gamma_ = self.gamma[i]

        # --- 2. Correction term ---
        eta = output + gamma_ * (output - u)

        # Format shape for network input
        eta = self.format_data(eta, dir=1)

        #Time embedding
        t = eta.new_tensor([(i+1)] * eta.shape[0]).long()
        u_next = self.R(eta, t)
        
        # Un-format shape for network input
        u_next = self.format_data(u_next, dir=-1)

        return u_next, gamma_


    #Loop DC and CNN blocks, with checkpointing
    for i in range(self.conf.nb_unroll_blocks + 1):

        # ======= DC BLOCK =======
        if self.conf.checkpoint and not self.test and cp[2 * i]:
            output, mu, x0 = checkpoint(dc_block, u, x0, use_reentrant=False)
        else:
            output, mu, x0 = dc_block(u, x0)

        # --- Exit on last block ---
        if i == self.conf.nb_unroll_blocks:
            if self.test: 
                return output, mu.item(), CNN_inputs, CNN_outputs
            else:
                return output, mu.item()

        CNN_inputs.append(output.detach())

        # ======= REGULARIZER (CNN) BLOCK =======
        if self.conf.checkpoint and not self.test and cp[2 * i + 1]:
            u, gamma_ = checkpoint(cnn_block, output, u, i, use_reentrant=False)
        else:
            u, gamma_ = cnn_block(output, u, i)

        gamma.append(gamma_)

        CNN_outputs.append(u.detach())

    return output, mu.item()

