import torch.nn as nn


class MLP(nn.Module):
    """
    multilayer perceptron (mlp) using pytorch.
    architecture: 784 -> 256 -> 128 -> 10 with relu activations.
    """

    def __init__(self, input_size=784, hidden1=256, hidden2=128, num_classes=10):
        """
        args:
            input_size: number of input features (784 for flattened mnist)
            hidden1: size of first hidden layer
            hidden2: size of second hidden layer
            num_classes: number of output classes (10 digits)
        """
        super(MLP, self).__init__()

        # define layers
        self.fc1 = nn.Linear(input_size, hidden1)  # 784 -> 256
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden1, hidden2)  # 256 -> 128
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden2, num_classes)  # 128 -> 10

    def forward(self, x):
        """
        forward pass through the network.

        args:
            x: input tensor, shape (batch_size, 784)

        returns:
            output scores, shape (batch_size, 10)
        """
        # pass through first hidden layer with relu
        x = self.fc1(x)
        x = self.relu1(x)

        # pass through second hidden layer with relu
        x = self.fc2(x)
        x = self.relu2(x)

        # pass through output layer (no activation, handled by loss function)
        x = self.fc3(x)

        return x
