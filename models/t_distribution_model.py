import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_model import BaseModel

class TModel(BaseModel):
    def __init__(self, input_size, hidden_size, num_hidden_layers, dropout_rate, activation_function):
        super(TModel, self).__init__(input_size, hidden_size, num_hidden_layers, dropout_rate, activation_function)
        self.output_layer = nn.Linear(hidden_size, 3)  # Output three parameters: mu, sigma, and raw nu

    def forward(self, x):
        x = super(TModel, self).forward(x)
        params = self.output_layer(x)
        mu = params[:, 0]
        sigma = F.softplus(params[:, 1])  # Ensure sigma is positive
        nu = F.softplus(params[:, 2]) + 2  # Ensure nu is positive and greater than 2
        return mu, sigma, nu
