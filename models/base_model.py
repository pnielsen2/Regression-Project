import torch
import torch.nn as nn

class BaseModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_hidden_layers, dropout_rate, activation_function):
        super(BaseModel, self).__init__()
        self.input_layer = nn.Linear(input_size, hidden_size)
        self.hidden_layers = nn.ModuleList([
            nn.Linear(hidden_size, hidden_size) for _ in range(num_hidden_layers)
        ])
        self.output_layer = None  # To be defined in derived classes
        self.batch_norm = nn.BatchNorm1d(hidden_size)
        self.dropout = nn.Dropout(dropout_rate)
        self.activation_function = self.get_activation_function(activation_function)
    
    def get_activation_function(self, name):
        if name == 'relu':
            return nn.ReLU()
        elif name == 'tanh':
            return nn.Tanh()
        elif name == 'leaky_relu':
            return nn.LeakyReLU()
        elif name == 'elu':
            return nn.ELU()
        elif name == 'gelu':
            return nn.GELU()
        elif name == 'selu':
            return nn.SELU()
        elif name == 'swish':
            return nn.SiLU()  # Swish is equivalent to SiLU in PyTorch
        elif name == 'mish':
            return nn.Mish()
        else:
            raise ValueError(f"Unsupported activation function: {name}")
    
    def forward(self, x):
        x = self.activation_function(self.input_layer(x))
        for layer in self.hidden_layers:
            x = self.activation_function(layer(self.batch_norm(x)))
            x = self.dropout(x)
        return x
