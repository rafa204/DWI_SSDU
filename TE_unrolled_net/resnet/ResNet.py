import torch
import torch.nn as nn
from configs import Config

conf = Config().parse()

class Residual_Block(nn.Module):
    def __init__(self):
        super(Residual_Block, self).__init__()
        
        self.conv = nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size=3, stride=1, padding=1, bias=False).cuda()
        self.relu = nn.ReLU(inplace=True)
        self.scalar = nn.Parameter(torch.tensor(0.1), requires_grad=False).cuda()

    def forward(self, inp):
        x = self.conv(inp)
        x = self.relu(x)        
        x = self.conv(x)
        x = self.scalar*x
        return x + inp

class ResNet(nn.Module):
    def __init__(self, nb_res_blocks):
        super(ResNet, self).__init__()
        in_channels = 2
        if conf.slice_chan:
            in_channels *= conf.mb

        self.first_layer = nn.Conv2d(in_channels = 2, out_channels = 64, kernel_size=3, stride=1, padding=1, bias=False)
        
        res_block = [] 
        for _ in range(nb_res_blocks):
            res_block += [Residual_Block()]
            
        self.res_block = nn.Sequential(*res_block)

        self.last_layer = nn.Conv2d(in_channels = 64, out_channels = 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.final_conv = nn.Conv2d(in_channels = 64, out_channels = 2, kernel_size=3, stride=1, padding=1, bias=False)
        self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight.data, mean=0.0, std=0.05)

    def forward(self, input_data):

        z = self.first_layer(input_data)
        output = self.res_block(z)
        output = self.last_layer(output)
        output = output + z
        output = self.final_conv(output)
        
        return output

