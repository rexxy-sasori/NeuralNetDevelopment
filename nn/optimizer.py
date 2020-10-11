"""
Author: Rex Geng

class definition for optimizer
"""

from torch import optim

from nn import quantization

__OPTIMIZER__ = {
    'sgd': optim.SGD,
    'adam': optim.Adam,
    'qsgd': quantization.QSGD,
    'qadam': quantization.QAdam,
}
