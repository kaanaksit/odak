import sys
from odak.learn.tools.matrix import generate_2d_dirac_delta


def test():
    dirac_delta = generate_2d_dirac_delta(normalize=True, a=[0.1, 0.1])
    assert dirac_delta[10][10] == 1., "The Dirac delta fucntion does not approximate peak correctly"

if __name__ == '__main__':
    sys.exit(test())
