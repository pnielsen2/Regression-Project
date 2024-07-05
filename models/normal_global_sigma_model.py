import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_model import BaseModel

class NormalModelGlobalSigma(BaseModel):
    def __init__(self, input_size, hidden_size, num_hidden_layers, dropout_rate, activation_function):
        super(NormalModelGlobalSigma, self).__init__(input_size, hidden_size, num_hidden_layers, dropout_rate, activation_function)
        self.output_layer = nn.Linear(hidden_size, 1)  # Output one parameter: mu
        self.sigma = nn.Parameter(torch.tensor([1.0]))  # Learnable global sigma

    def forward(self, x):
        x = super(NormalModelGlobalSigma, self).forward(x)
        mu = self.output_layer(x).squeeze()
        sigma = F.softplus(self.sigma)  # Ensure sigma is positive
        return mu, sigma
