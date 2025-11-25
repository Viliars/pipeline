

import math
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from .TCNBlock import TCNBlock
import torch.nn.init as winit
from .separableConv1d import separableConv1d

class TCN(nn.Module):
  def __init__(self, num_inputs, num_channels, kernel_size=3, repeat=[1,2,5,9]*4):
    super(TCN, self).__init__()
    layers = []
    num_levels = len(num_channels)
    for i in range(num_levels):
        d1 = repeat[2*i] 
        d2 = repeat[2*i+1] 
        in_channels = num_inputs if i == 0 else num_channels[i-1]
        out_channels = num_channels[i]
        layers += [TCNBlock(in_channels, out_channels, kernel_size, 1, d1,d2)]

    self.network = nn.Sequential(*layers)


  def forward(self, x):
        return self.network(x)
