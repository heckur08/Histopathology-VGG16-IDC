# Import all the necessary libraries
# Import torch libraries
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

# Import non-deeplearning libraries
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2

# # Model definition
# The base paper for the VGG net can be found [here](https://arxiv.org/pdf/1409.1556.pdf)
class VGG16(nn.Module):
    def __init__(self, hidden = 64, no_of_classes=2, kernel_size=(3,3), padding=1, stride=(1,1)):
        super(VGG16, self).__init__()
        '''
        Initializes the layers for the neral network model. The VGG-16 with Batch Normalization is used
        
        Arguments:
        hidden  --  The number of channels for the first layer. The no of channels for the consequent layers are a multiple of this number
        no_of_classes --  The number of output classes
        kernel_size  --  Filter Size for the Conv layers
        padding  --  Padding for the Conv layers. For SAME conv, padding = (kernel_size - 1)/2
        stride  --  Stride for the Conv layers
        
        Return:
        None
        '''
        # Input Shape is 64x64
        # Block 1. It consists of the Conv layers till the first max pooling layer
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=hidden, kernel_size=kernel_size, padding=padding, stride=stride)
        self.bn1 = nn.BatchNorm2d(num_features = hidden)
        self.conv2 = nn.Conv2d(in_channels=hidden, out_channels=hidden, kernel_size=kernel_size, padding=padding, stride=stride)
        self.bn2 = nn.BatchNorm2d(num_features = hidden)
        self.max1  = nn.MaxPool2d(kernel_size = (2,2), stride = 2)

        # Block 2. It consists of the Conv layers till the second max pooling layer
        self.conv3 = nn.Conv2d(in_channels=hidden, out_channels=2*hidden, kernel_size=kernel_size, padding=padding, stride=stride)
        self.bn3 = nn.BatchNorm2d(num_features = 2*hidden)
        self.conv4 = nn.Conv2d(in_channels=2*hidden, out_channels=2*hidden, kernel_size=kernel_size, padding=padding, stride=stride)
        self.bn4 = nn.BatchNorm2d(num_features = 2*hidden)
        self.max2  = nn.MaxPool2d(kernel_size = (2,2), stride = 2)

        # Block 3. It consists of the Conv layers till the third max pooling layer
        self.conv5 = nn.Conv2d(in_channels=2*hidden, out_channels=4*hidden, kernel_size=kernel_size, padding=padding, stride=stride)
        self.bn5 = nn.BatchNorm2d(num_features = 4*hidden)
        self.conv6 = nn.Conv2d(in_channels=4*hidden, out_channels=4*hidden, kernel_size=kernel_size, padding=padding, stride=stride)
        self.bn6 = nn.BatchNorm2d(num_features = 4*hidden)
        self.conv7 = nn.Conv2d(in_channels=4*hidden, out_channels=4*hidden, kernel_size=kernel_size, padding=padding, stride=stride)
        self.bn7 = nn.BatchNorm2d(num_features = 4*hidden)
        self.max3  = nn.MaxPool2d(kernel_size = (2,2), stride = 2)

        # Block 4. It consists of the Conv layers till the fourth max pooling layer
        self.conv8  = nn.Conv2d(in_channels=4*hidden, out_channels=8*hidden, kernel_size=kernel_size, padding=padding, stride=stride)
        self.bn8 = nn.BatchNorm2d(num_features = 8*hidden)
        self.conv9  = nn.Conv2d(in_channels=8*hidden, out_channels=8*hidden, kernel_size=kernel_size, padding=padding, stride=stride)
        self.bn9 = nn.BatchNorm2d(num_features = 8*hidden)
        self.conv10 = nn.Conv2d(in_channels=8*hidden, out_channels=8*hidden, kernel_size=kernel_size, padding=padding, stride=stride)
        self.bn10 = nn.BatchNorm2d(num_features = 8*hidden)
        self.max4  = nn.MaxPool2d(kernel_size = (2,2), stride = 2)

        # Block 5. It consists of the Conv layers till the fifth max pooling layer
        self.conv11 = nn.Conv2d(in_channels=8*hidden, out_channels=8*hidden, kernel_size=kernel_size, padding=padding, stride=stride)
        self.bn11 = nn.BatchNorm2d(num_features = 8*hidden)
        self.conv12 = nn.Conv2d(in_channels=8*hidden, out_channels=8*hidden, kernel_size=kernel_size, padding=padding, stride=stride)
        self.bn12 = nn.BatchNorm2d(num_features = 8*hidden)
        self.conv13 = nn.Conv2d(in_channels=8*hidden, out_channels=8*hidden, kernel_size=kernel_size, padding=padding, stride=stride)
        self.bn13 = nn.BatchNorm2d(num_features = 8*hidden)
        self.max5  = nn.MaxPool2d(kernel_size = (2,2), stride = 2)

        # Fully connected layers and Classification
        # in_features = dim x dim x 512
        # dim = input_shape / 32. Dimensions after the 32x downsampling
        self.fc1 = nn.Linear(in_features=2*2*512, out_features = 4096)
        self.fc2 = nn.Linear(in_features=4096, out_features = 4096)
        self.final_layer = nn.Linear(in_features=4096, out_features=no_of_classes)

    def forward(self, x):
        '''
        Forward propogation of the model. The ReLU activations are used for the Conv layers
        
        Arguments:
        x  --  Input to the VGG network
        
        Returns
        y  --  Output of the VGG network
        '''
        # Block 1
        x_1 = nn.ReLU()(self.bn1(self.conv1(x)))
        x_1 = nn.ReLU()(self.bn2(self.conv2(x_1)))
        x_1 = self.max1(x_1)

        # Block 2
        x_2 = nn.ReLU()(self.bn3(self.conv3(x_1)))
        x_2 = nn.ReLU()(self.bn4(self.conv4(x_2)))
        x_2 = self.max2(x_2)

        # Block 3
        x_3 = nn.ReLU()(self.bn5(self.conv5(x_2)))
        x_3 = nn.ReLU()(self.bn6(self.conv6(x_3)))
        x_3 = nn.ReLU()(self.bn7(self.conv7(x_3)))
        x_3 = self.max3(x_3)

        # Block 4
        x_4 = nn.ReLU()(self.bn8(self.conv8(x_3)))
        x_4 = nn.ReLU()(self.bn9(self.conv9(x_4)))
        x_4 = nn.ReLU()(self.bn10(self.conv10(x_4)))
        x_4 = self.max4(x_4)

        # Block 5
        x_5 = nn.ReLU()(self.bn11(self.conv11(x_4)))
        x_5 = nn.ReLU()(self.bn12(self.conv12(x_5)))
        x_5 = nn.ReLU()(self.bn13(self.conv12(x_5)))
        x_5 = self.max5(x_5)

        # The 32x downsampled image is flattened and passed through Linear and Classification layers
        x_flat = nn.Flatten()(x_5)
        x_fc = self.fc1(x_flat)
        x_fc = self.fc2(x_fc)
        y = self.final_layer(x_fc)
        return y