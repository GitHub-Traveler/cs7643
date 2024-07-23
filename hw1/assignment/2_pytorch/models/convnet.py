import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self, im_size, hidden_dim, kernel_size, n_classes):
        '''
        Create components of a CNN classifier and initialize their weights.

        Arguments:
            im_size (tuple): A tuple of ints with (channels, height, width)
            hidden_dim (int): Number of hidden activations to use
            kernel_size (int): Width and height of (square) convolution filters
            n_classes (int): Number of classes to score
        '''
        super(CNN, self).__init__()
        #############################################################################
        # TODO: Initialize anything you need for the forward pass
        #############################################################################
        in_channels, height, width = im_size
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size, 2, 1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size, 2, 1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size, 1, 1)
        self.bn3 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.linear1 = nn.Linear(4096, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, n_classes)
        pass
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################

    def forward(self, images):
        '''
        Take a batch of images and run them through the CNN to
        produce a score for each class.

        Arguments:
            images (Variable): A tensor of size (N, C, H, W) where
                N is the batch size
                C is the number of channels
                H is the image height
                W is the image width

        Returns:
            A torch Variable of size (N, n_classes) specifying the score
            for each example and category.
        '''
        scores = None
        #############################################################################
        # TODO: Implement the forward pass. This should take few lines of code.
        #############################################################################
        hidden = self.conv1(images)
        # print('cailonma')
        hidden = self.bn1(hidden)
        hidden = self.relu(hidden)
        hidden = self.conv2(hidden)
        hidden = self.bn2(hidden)
        hidden = self.relu(hidden)
        hidden = self.conv3(hidden)
        hidden = self.bn3(hidden)
        hidden = self.relu(hidden)
        hidden = torch.flatten(hidden, start_dim = 1)
        hidden = self.linear1(hidden)
        hidden = self.relu(hidden)
        scores = self.linear2(hidden)
        pass
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
        return scores

