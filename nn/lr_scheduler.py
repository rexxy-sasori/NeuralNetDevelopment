"""
Author: Rex Geng

class definition for learning rate scheduler
"""
from torch import optim

__SCHEDULER__ = {
    'step': optim.lr_scheduler.StepLR,
    'rdp': optim.lr_scheduler.ReduceLROnPlateau
}
