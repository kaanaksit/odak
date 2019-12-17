#!/usr/bin/env python
#
# This allows running the tests for odak:
#
#   $ python tests.py
#
# See help for more arguments.
#
#   $ pyton test.py --help
#

import sys
import argparse

def test_odak_import():
    import odak
    assert True=True

if __name__ == '__main__':

    
    sys.exit(test_odak_import())
