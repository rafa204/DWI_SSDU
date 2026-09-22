from torch.utils.checkpoint import checkpoint
import sys

def vsqp_unrolling(self, ksp, coil, mask, conf):

    zerofilled = self.dc.EH(ksp, coil, mask)

    if conf.warm_start:
        recon, mu = self.dc(zerofilled, coil, mask)
    else:
        recon = zerofilled

    output = recon.clone()
    reg_input = []
    reg_output = []
    x0 = None

    #### DEFINE A SINGLE UNROLL BLOCK
    def unroll_block(*args):

        output, i = args

        reg_input.append(output.detach().cpu())
        output = self.format_data(output, dir = 1)

        # --- 1. Regularizer ---
        if "time_emb" in conf.model:
            t = output.new_tensor([i+1] * output.shape[0]).long().to(output.device)
            output = self.R(output, t)  # t = i
        elif "multi" in conf.model:
            output = self.R[i](output)
        else:
            output = self.R(output)
        output = self.format_data(output, dir = -1)
        reg_output.append(output.detach().cpu())

        # --- 2. Data Fidelity ---
        if conf.DC_unshared:
            output, mu_ = self.dc[i](zerofilled, coil, mask, output, x0 = x0)
            mu.append(mu_)
        else:
            output, mu = self.dc(zerofilled, coil, mask, output, x0 = x0)

        if conf.warm_start:
            x0 = output.clone()
        else:
            x0 = None
        
        if self.test:
            return output, mu.item(), reg_input, reg_output
        else:
            return output, mu.item()
    #### 

    for i in range(conf.nb_unroll_blocks):
        if conf.checkpoint and not self.test:
            output_list = checkpoint(unroll_block, output, i, use_reentrant=False)
        else:
            output_list = unroll_block(output, i)
        output = output_list[0]

    return output_list

