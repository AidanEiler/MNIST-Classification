import torch.nn as nn


class CNN(nn.Module):
    """
    convolutional neural network (cnn) using pytorch.
    architecture: conv(1->32, 3x3) -> relu -> maxpool -> conv(32->64, 3x3) -> relu -> maxpool -> fc -> softmax
    meets requirements: at least two convolutional layers with activation + pooling
    """

    def __init__(self, num_classes=10):
        """
        args:
            num_classes: number of output classes (10 digits)
        """
        super(CNN, self).__init__()

        # first convolutional layer: 1->32 channels, 3x3 kernel
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # 28x28 -> 14x14

        # second convolutional layer: 32->64 channels, 3x3 kernel
        self.conv2 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=3, padding=1
        )
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # 14x14 -> 7x7

        # fully connected layer
        self.fc = nn.Linear(
            64 * 7 * 7, num_classes
        )  # 7x7 feature maps, 64 channels -> 10 classes

        # softmax handled by crossentropyloss in training (includes log softmax)

    def forward(self, x):
        """
        forward pass through the network.

        args:
            x: input tensor, shape (batch_size, 1, 28, 28)

        returns:
            output scores, shape (batch_size, 10)
        """
        # first conv layer with activation and pooling
        x = self.conv1(x)  # (batch, 1, 28, 28) -> (batch, 32, 28, 28)
        x = self.relu1(x)  # activation
        x = self.pool1(x)  # (batch, 32, 28, 28) -> (batch, 32, 14, 14)

        # second conv layer with activation and pooling
        x = self.conv2(x)  # (batch, 32, 14, 14) -> (batch, 64, 14, 14)
        x = self.relu2(x)  # activation
        x = self.pool2(x)  # (batch, 64, 14, 14) -> (batch, 64, 7, 7)

        # flatten for fully connected layer
        x = x.view(x.size(0), -1)  # (batch, 64, 7, 7) -> (batch, 3136)

        # fully connected layer (softmax applied by loss function)
        x = self.fc(x)  # (batch, 3136) -> (batch, 10)

        return x
