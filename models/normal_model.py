import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_model import BaseModel

class NormalModel(BaseModel):
    def __init__(self, input_size, hidden_size, num_hidden_layers, dropout_rate, activation_function):
        super(NormalModel, self).__init__(input_size, hidden_size, num_hidden_layers, dropout_rate, activation_function)
        self.output_layer = nn.Linear(hidden_size, 2)  # Output two parameters: mu and sigma

    def forward(self, x):
        x = super(NormalModel, self).forward(x)
        params = self.output_layer(x)
        theta_1, theta_2 =  params[:, 0], params[:, 1]
        mu, sigma = -theta_1/(2*theta_2), torch.sqrt(torch.abs(-1/(2*theta_2)))
        # sigma = F.softplus(params[:, 1])  # Ensure sigma is positive
        return mu, sigma
