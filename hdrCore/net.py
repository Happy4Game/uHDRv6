# uHDR: HDR image editing software
#   Copyright (C) 2021  remi cozot 
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
# hdrCore project 2020
# author: remi.cozot@univ-littoral.fr

# -----------------------------------------------------------------------------
# --- Package hdrCore ---------------------------------------------------------
# -----------------------------------------------------------------------------
"""
HDR Core Neural Network Module

This module provides PyTorch-based neural network architectures for HDR image
processing tasks. It includes custom network definitions and forward pass
implementations optimized for HDR imaging applications.

Classes:
    Net: Simple feedforward neural network with batch normalization
"""

# -----------------------------------------------------------------------------
# --- Import ------------------------------------------------------------------
# -----------------------------------------------------------------------------
import torch
import torch.nn as nn

# -----------------------------------------------------------------------------
# --- class Net ---------------------------------------------------------------
# -----------------------------------------------------------------------------
class Net(nn.Module):
    """
    Simple feedforward neural network for HDR processing tasks.
    
    A compact neural network architecture with batch normalization and sigmoid
    activation, designed for feature extraction and processing in HDR imaging
    workflows. The network includes a single hidden layer with batch normalization
    for stable training.
    
    Attributes:
        - layer (nn.Sequential): Sequential layer containing linear transformation,
                              batch normalization, and sigmoid activation
    
    Args:
        - n_feature (int): Number of input features
        - n_output (int): Number of output neurons (currently not used in implementation)
        
    Note:
        The current implementation uses a fixed architecture with 5 hidden units
        regardless of the n_output parameter.
    """
    
    def __init__(self, n_feature, n_output):
        """
        Initialize the neural network with specified input and output dimensions.
        
        Args:
            n_feature (int): Number of input features for the network
            n_output (int): Number of output neurons (for future extensibility)
        """
        super(Net, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(n_feature, 5),
            nn.BatchNorm1d(5),
            nn.Sigmoid(),
        )
    
    # -----------------------------------------------------------------------------
    def forward(self, x):
        """
        Forward pass through the neural network.
        
        Processes input data through the linear layer, batch normalization,
        and sigmoid activation function.
        
        Args:
            x (torch.Tensor): Input tensor with shape (batch_size, n_feature)
            
        Returns:
            torch.Tensor: Output tensor with shape (batch_size, 5) containing
                         processed features with values in range [0, 1] due to
                         sigmoid activation
        """
        return self.layer(x)
    