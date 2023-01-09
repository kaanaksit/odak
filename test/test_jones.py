#!/usr/bin/env python

import sys


def test():
    import odak.jones as jones
    u_in = jones.electricfield(0.5, 0.)
    u_out = jones.linearpolarizer(u_in, rotation=0)
    print(u_out)
    assert True == True


if __name__ == '__main__':
    sys.exit(test())
