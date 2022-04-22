import os
import torch
import numpy as np
from dataloader import CustomDataset
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from model_blocks import conv, deconv, ResidualBlock


class CycleGenerator(nn.Module):
    
    def __init__(self, conv_dim=64, n_res_blocks=6):
        super(CycleGenerator, self).__init__()

        # 1. Define the encoder part of the generator
        self.conv1 = conv(3, conv_dim, 4)
        self.conv2 = conv(conv_dim, conv_dim*2, 4)
        self.conv3 =conv(conv_dim*2, conv_dim*4, 4)

        # 2. Define the resnet part of the generator
        layers = []
        for layer in range(n_res_blocks):
            layers.append(ResidualBlock(conv_dim*4))
        # use sequential to create these layers
        self.res_blocks = nn.Sequential(*layers)


        # 3. Define the decoder part of the generator
        self.deconv1 = deconv(conv_dim*4, conv_dim*2, 4)
        self.deconv2 = deconv(conv_dim*2, conv_dim, 4)
        # no batch norm on last layer
        self.deconv3 = deconv(conv_dim, 3, 4, batch_norm=False)

        # Pointcloud support block
        fc_input_dims = self.calculate_conv_output_dims([3,224,224])
        self.flatten = nn.Flatten()

        # Linear layers output is the same as pointcloud output 1,5000,3
        self.fc1 = nn.Linear(fc_input_dims, 1000)
        self.fc2 = nn.Linear(1000, 5000) 
        self.fc3 = nn.Linear(5000, 15000)



    def calculate_conv_output_dims(self, input_dims):
        state = torch.zeros(1, *input_dims)
        dims = self.conv1(state)
        dims = self.conv2(dims)
        dims = self.conv3(dims)

        dims = self.res_blocks(dims)
        
        dims = F.relu(self.deconv1(dims))
        dims = F.relu(self.deconv2(dims))
        dims = F.relu(self.deconv3(dims))
        # print(int(np.prod(dims.size())))
        return int(np.prod(dims.size()))

    def forward(self, x, batch_size):
        """Given an image x, returns a transformed image."""
        # define feedforward behavior, applying activations as necessary
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        print(x.size())

        x = self.res_blocks(x)
        
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = F.relu(self.deconv3(x))

        # Get the forward prop result and return the reshaped version
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        print(x.size())

        # Reshape as batch size + pointcloud shape[5000,3]
        x = torch.reshape(x, [batch_size,5000,3])

        return x

    


