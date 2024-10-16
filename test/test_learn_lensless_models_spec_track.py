import sys
sys.path.append("./")
import torch
import odak
import numpy as np
from odak.learn.lensless import spec_track


def test():
    x = torch.randn(96, 5, 180, 320)
    network = spec_track()
    assert network(x).shape == (96, 3)
    
if __name__ == '__main__':
    sys.exit(test())