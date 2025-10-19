import torch.nn as nn


class LinearClassifier(nn.Module):
    """
    linear classifier using pytorch.
    implements y = wx + b where w and b are learned through gradient descent.
    """

    def __init__(self, input_size=784, num_classes=10):
        """
        args:
            input_size: number of input features (784 for flattened mnist)
            num_classes: number of output classes (10 digits)
        """
        super(LinearClassifier, self).__init__()
        # single linear layer that does wx + b
        self.linear = nn.Linear(input_size, num_classes)

    def forward(self, x):
        """
        forward pass through the model.

        args:
            x: input tensor, shape (batch_size, 784)

        returns:
            output scores, shape (batch_size, 10)
        """
        # just apply the linear transformation
        return self.linear(x)