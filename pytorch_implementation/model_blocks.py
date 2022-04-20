import os


from dataloader import CustomDataset
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F


class conv_utils():
    def __init__(self) -> None:
        pass
    # helper conv function
    def conv(self, in_channels, out_channels, kernel_size, stride=2, padding=1, batch_norm=True):
        """Creates a convolutional layer, with optional batch normalization.
        """
        layers = []
        conv_layer = nn.Conv2d(in_channels, out_channels, 
                            kernel_size, stride, padding, bias=False)
        
        # append conv layer
        layers.append(conv_layer)

        if batch_norm:
            # append batchnorm layer
            layers.append(nn.BatchNorm2d(out_channels))
        
        # using Sequential container
        return nn.Sequential(*layers)


    # helper deconv function
    def deconv(self, in_channels, out_channels, kernel_size, stride=2, padding=1, batch_norm=True):
        """Creates a transposed-convolutional layer, with optional batch normalization."""
        # create a sequence of transpose + optional batch norm layers
        layers = []
        transpose_conv_layer = nn.ConvTranspose2d(in_channels, out_channels, 
                                                kernel_size, stride, padding, bias=False)
        # append transpose convolutional layer
        layers.append(transpose_conv_layer)
        
        if batch_norm:
            # append batchnorm layer
            layers.append(nn.BatchNorm2d(out_channels))
            
        return nn.Sequential(*layers)


    # residual block class
class ResidualBlock(nn.Module):
    """Defines a residual block.
       This adds an input x to a convolutional layer (applied to x) with the same size input and output.
       These blocks allow a model to learn an effective transformation from one domain to another.
    """
    def __init__(self, conv_dim):
        super(ResidualBlock, self).__init__()
        # conv_dim = number of inputs  
        
        # define two convolutional layers + batch normalization that will act as our residual function, F(x)
        # layers should have the same shape input as output; I suggest a kernel_size of 3

        self.conv1 = conv(3, conv_dim,
                                kernel_size=3, stride=1, padding=1, batch_norm=True)
        self.conv2 = conv(conv_dim, conv_dim,
                                kernel_size=3, stride=1, padding=1, batch_norm=True)
        
        
    def forward(self, x):
        # apply a ReLu activation the outputs of the first layer
        # return a summed output, x + resnet_block(x)
        out1 = F.relu(self.conv1(x))
        out2 = x+F.relu(self.conv2(x))
        x = out2
        
        return x
