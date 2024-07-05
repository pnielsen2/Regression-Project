import torch
import torch.nn as nn

class TDistributionLoss(nn.Module):
    def __init__(self):
        super(TDistributionLoss, self).__init__()

    def forward(self, x, mu, sigma, nu):
        term1 = torch.lgamma((nu + 1) / 2) - torch.lgamma(nu / 2)
        term2 = -0.5 * torch.log(nu * torch.pi * sigma ** 2)
        term3 = -(nu + 1) / 2 * torch.log(1 + (1 / nu) * ((x - mu) / sigma) ** 2)
        nll = -(term1 + term2 + term3).mean()
        return nll

class NormalDistributionLoss(nn.Module):
    def __init__(self):
        super(NormalDistributionLoss, self).__init__()

    def forward(self, x, mu, sigma):
        term1 = torch.log(sigma * torch.sqrt(torch.tensor(2 * torch.pi)))  # Fix here
        term2 = 0.5 * ((x - mu) / sigma) ** 2
        nll = (term1 + term2).mean()
        return nll
