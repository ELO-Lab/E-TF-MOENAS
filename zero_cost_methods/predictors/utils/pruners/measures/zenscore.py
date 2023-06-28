import torch
from torch import nn
import numpy as np
from . import measure

def network_weight_gaussian_init(net: nn.Module):
    with torch.no_grad():
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.zeros_(m.bias)
            else:
                continue
    return net

@measure("zenscore", bn=True, mode='')
def compute_zenscore(net, inputs, targets, mode, split_data=1, loss_fn=None):
    mixup_gamma = 1e-2
    device = inputs.device

    nas_score_list = []
    with torch.no_grad():
        network_weight_gaussian_init(net)
        input = torch.randn(size=[1, 3, 32, 32], device=device)
        input2 = torch.randn(size=[1, 3, 32, 32], device=device)
        mixup_input = input + mixup_gamma * input2
        output = net.forward_pre_GAP(input)
        mixup_output = net.forward_pre_GAP(mixup_input)

        nas_score = torch.sum(torch.abs(output - mixup_output), dim=[1, 2, 3])
        nas_score = torch.mean(nas_score)

        # compute BN scaling
        log_bn_scaling_factor = 0.0
        for m in net.modules():
            if isinstance(m, nn.BatchNorm2d):
                bn_scaling_factor = torch.sqrt(torch.mean(m.running_var))
                log_bn_scaling_factor += torch.log(bn_scaling_factor)
            pass
        pass
        nas_score = torch.log(nas_score) + log_bn_scaling_factor
        nas_score_list.append(float(nas_score))

    avg_nas_score = np.mean(nas_score_list)
    return float(avg_nas_score)