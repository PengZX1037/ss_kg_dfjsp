import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):

    def __init__(self, num_layers, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.linear_or_not = True
        self.num_layers = num_layers
        if num_layers < 1:
            raise ValueError('number of layers should be positive!')
        elif num_layers == 1:
            self.linear = nn.Linear(input_dim, output_dim)
        else:
            self.linear_or_not = False
            self.linears = torch.nn.ModuleList()
            self.linears.append(nn.Linear(input_dim, hidden_dim))
            for layer in range(num_layers - 2):
                self.linears.append(nn.Linear(hidden_dim, hidden_dim))
            self.linears.append(nn.Linear(hidden_dim, output_dim))

    def forward(self, x):
        if self.linear_or_not:
            return self.linear(x)
        else:
            hidden_features = x
            for layer_index in range(self.num_layers - 1):
                hidden_features = F.relu(self.linears[layer_index](hidden_features))
            return self.linears[self.num_layers - 1](hidden_features)

class Actor(nn.Module):

    def __init__(self, num_layers, input_dim, hidden_dim, output_dim):
        super(Actor, self).__init__()
        self.linear_or_not = True
        self.num_layers = num_layers
        self.activative = torch.tanh
        if num_layers < 1:
            raise ValueError('number of layers should be positive!')
        elif num_layers == 1:
            self.linear = nn.Linear(input_dim, output_dim)
        else:
            self.linear_or_not = False
            self.linears = torch.nn.ModuleList()
            self.linears.append(nn.Linear(input_dim, hidden_dim))
            for layer in range(num_layers - 2):
                self.linears.append(nn.Linear(hidden_dim, hidden_dim))
            self.linears.append(nn.Linear(hidden_dim, output_dim))

    def forward(self, x):
        if self.linear_or_not:
            return self.linear(x)
        else:
            hidden_features = x
            for layer_index in range(self.num_layers - 1):
                hidden_features = self.activative(self.linears[layer_index](hidden_features))
            return self.linears[self.num_layers - 1](hidden_features)

class Critic(nn.Module):

    def __init__(self, num_layers, input_dim, hidden_dim, output_dim):
        super(Critic, self).__init__()
        self.linear_or_not = True
        self.num_layers = num_layers
        self.activative = torch.tanh
        if num_layers < 1:
            raise ValueError('number of layers should be positive!')
        elif num_layers == 1:
            self.linear = nn.Linear(input_dim, output_dim)
        else:
            self.linear_or_not = False
            self.linears = torch.nn.ModuleList()
            self.linears.append(nn.Linear(input_dim, hidden_dim))
            for layer in range(num_layers - 2):
                self.linears.append(nn.Linear(hidden_dim, hidden_dim))
            self.linears.append(nn.Linear(hidden_dim, output_dim))

    def forward(self, x):
        if self.linear_or_not:
            return self.linear(x)
        else:
            hidden_features = x
            for layer_index in range(self.num_layers - 1):
                hidden_features = self.activative(self.linears[layer_index](hidden_features))
            return self.linears[self.num_layers - 1](hidden_features)
