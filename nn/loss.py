"""
Author: Rex Geng

class definition for loss functions
"""

from torch import nn

__LOSS__ = {
    'cel': nn.CrossEntropyLoss,
    'wcel': nn.CrossEntropyLoss,
    'mse': nn.MSELoss
}
