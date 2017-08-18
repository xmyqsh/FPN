# --------------------------------------------------------
# SubCNN_TF
# Copyright (c) 2016 CVGL Stanford
# Licensed under The MIT License [see LICENSE for details]
# Written by Yu Xiang
# --------------------------------------------------------

"""Factory method for easily getting imdbs by name."""

__sets = {}

from .FPN_train import FPN_train
from .FPN_test import FPN_test
from .FPN_alt_opt_train import FPN_alt_opt_train

def get_network(name):
    """Get a network by name."""
    if name.split('_')[0] == 'FPN':
        if name.split('_')[1] == 'test':
           return FPN_test()
        elif name.split('_')[1] == 'train':
           return FPN_train()
        elif name.split('_')[1] == 'alt':
            if name.split('_')[3] == 'train':
                return FPN_alt_opt_train()
            else:
                raise KeyError('Unknown dataset: {}'.format(name))
        else:
           raise KeyError('Unknown dataset: {}'.format(name))
    else:
        raise KeyError('Unknown dataset: {}'.format(name))

def list_networks():
    """List all registered imdbs."""
    return __sets.keys()
