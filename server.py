import flwr as fl
import torch.nn as nn
import torch.optim as optim

class LinearRegressionModel(nn.Module):
    def __init__(self):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(1, 1)

# Define a simple strategy
strategy = fl.server.strategy.FedAvg(fraction_fit=1.0, min_available_clients=2)

# Create Flower server
fl.server.start_server(num_rounds=3, strategy=strategy)
