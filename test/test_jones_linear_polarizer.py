#!/usr/bin/env python

import sys
import odak.jones as jones


def test():
    u_in = jones.electricfield(0.5, 0.)
    u_out = jones.linearpolarizer(u_in, rotation=0)
    assert True == True


if __name__ == '__main__':
    sys.exit(test())
