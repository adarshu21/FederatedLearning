import flwr as fl
import torch
import torch.nn as nn
import torch.optim as optim

class LinearRegressionModel(nn.Module):
    def __init__(self):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(1, 1)

# Load the dataset (replace this with your actual data loading logic)
# For simplicity, let's assume X_train and y_train are loaded from your dataset
X_train = torch.tensor([[23], [3], [49]], dtype=torch.float32)
y_train = torch.tensor([[19114.12], [1824.84], [49.57]], dtype=torch.float32)

# Define a simple client
class LinearRegressionClient(fl.client.NumPyClient):
    def __init__(self, model):
        self.model = model
        self.criterion = nn.MSELoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.01)

    def get_parameters(self):
        return [param.numpy() for param in self.model.parameters()]

    def set_parameters(self, parameters):
        self.model.set_parameters(parameters)

    def fit(self, parameters, config):
        self.model.set_parameters(parameters)
        predictions = self.model(X_train)
        loss = self.criterion(predictions, y_train)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return self.get_parameters(), len(X_train), {"loss": loss.item()}

    def evaluate(self, parameters, config):
        self.model.set_parameters(parameters)
        predictions = self.model(X_train)
        loss = self.criterion(predictions, y_train)

        return loss.item(), len(X_train), {"loss": loss.item()}

# Create Flower client
fl.client.start_numpy_client("[::]:8080", client=LinearRegressionClient(model=LinearRegressionModel()))
