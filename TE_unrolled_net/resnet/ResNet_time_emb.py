import torch
import torch.nn as nn
from configs import Config
from TE_unrolled_net.unet.nn import timestep_embedding, linear
# from utils import set_global_seed

# set_global_seed(1)
conf = Config().parse()

class Residual_Block(nn.Module):
    def __init__(self):
        super(Residual_Block, self).__init__()
        self.conv = nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size=3, stride=1, padding=1, bias=False).cuda()
        self.relu = nn.ReLU(inplace=True)
        self.scalar = nn.Parameter(torch.tensor(0.1), requires_grad=False).cuda()

        emb_channels = 64 * conf.time_emb_ratio
        self.emb_layers = nn.Sequential(
                        nn.SiLU(),
                        linear(emb_channels, 2 * 64)
                        )

        self.silu = nn.SiLU(inplace=True)
        self.normalization = nn.GroupNorm(32, 64)
        self.tau = conf.tau_init  # conf.tau_init = default 0.1

    def forward(self, inp, emb):
        emb_out = self.emb_layers(emb)  
        while len(emb_out.shape) < len(inp.shape):
            emb_out = emb_out[..., None]              

        x = self.conv(inp)
        x = self.relu(x)
        x = self.conv(x)

        x_ = self.normalization(x)
        scale, shift = torch.chunk(emb_out, 2, dim=1)
        x_te = x_ * scale + shift           
        x = x + self.tau * x_te    

        x = self.silu(x)
        x = self.scalar*x

        return x + inp


class ResNet_time_emb(nn.Module):
    def __init__(self, nb_res_blocks, num_channels = 2):
        super(ResNet_time_emb, self).__init__()
        self.first_layer = nn.Conv2d(in_channels = num_channels, out_channels = 64, kernel_size=3, stride=1, padding=1, bias=False)

        self.nb_res_blocks = nb_res_blocks

        res_block = []  
        for _ in range(nb_res_blocks):
            res_block += [Residual_Block()]        
        self.res_block = nn.Sequential(*res_block)

        self.last_layer = nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.final_conv = nn.Conv2d(in_channels = 64, out_channels = num_channels, kernel_size=3, stride=1, padding=1, bias=False)

        self.time_dim = 64
        time_embed_dim = self.time_dim * conf.time_emb_ratio
        self.time_embed = nn.Sequential(
            nn.Linear(self.time_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

        self.initialize_weights()


    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight.data, mean=0.0, std=0.05)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight.data, mean=0.0, std=0.05)


    def forward(self, input_data, timesteps):

        emb = self.time_embed(timestep_embedding(timesteps.to(dtype=input_data.dtype), self.time_dim))
        z = self.first_layer(input_data)

        output = z

        for i in range(self.nb_res_blocks):
            output = self.res_block[i](output, emb)

        output = self.last_layer(output)
        output = output + z
        output = self.final_conv(output)


        return output

