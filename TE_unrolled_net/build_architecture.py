import torch
import torch.nn as nn
from TE_unrolled_net.resnet.ResNet import ResNet
from TE_unrolled_net.resnet.ResNet_time_emb import ResNet_time_emb
from TE_unrolled_net.resnet.ResNet_time_emb_3D import ResNet_time_emb3D
from configs import Config
conf = Config().parse()

if conf.spirit:
    from DC_SPIRIT import DC_SPIRIT as Data_consistency
else:
    from DC_SENSE import DC_SENSE as Data_consistency

def build_regularizer(conf):
    n_channels = 2
    if "ResNet_time_emb" in conf.model:
        if conf.qdim:
            return ResNet_time_emb3D(conf.nb_res_blocks, n_channels)
        else:
            return ResNet_time_emb(conf.nb_res_blocks, n_channels)

    else:
        raise ValueError(f"Unknown regularizer type: {conf.model}")

def build_dc(conf):
    if conf.DC_unshared:
        blocks = conf.nb_unroll_blocks + 1 if conf.Unroll_algo in ["ADMM", "VAMP"] else conf.nb_unroll_blocks
        return nn.ModuleList([Data_consistency(mu=conf.mu_init) for _ in range(blocks)])
    else:
        return Data_consistency(mu=conf.mu_init)

def build_gamma(conf):
    if conf.Unroll_algo not in ["ADMM", "VAMP"]:
        return None

    if conf.gamma_unshared:
        return nn.ParameterList([
            nn.Parameter(torch.tensor(conf.gamma_init), requires_grad=True)
            for _ in range(conf.nb_unroll_blocks)
        ])
    else:
        return nn.Parameter(torch.tensor(conf.gamma_init), requires_grad=True)
