
import torch
import torch.nn as nn
import numpy as np
import os

class network(nn.Module):
    def __init__(self, acc_rate, no_ch, padding = 'valid'):
        super(network,self).__init__()
        self.R = acc_rate
        self.channels_in = no_ch
        self.net = nn.Sequential( 
            nn.Conv2d(in_channels= no_ch, out_channels=32, kernel_size=[2, 5], dilation= (acc_rate,1),padding=padding,bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=8, kernel_size=[1, 1], padding=padding, dilation= (acc_rate,1), bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels=8, out_channels=(self.R-1), kernel_size=[2, 3],dilation= (acc_rate,1), padding=padding, bias=False),           
        )


    def forward(self, x):
        return self.net(x)
        

