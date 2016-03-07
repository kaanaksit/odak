#!/usr/bin/python
# -*- coding: utf-8 -*-

# Script taken from http://stackoverflow.com/questions/34803197/fast-b-spline-algorithm-with-numpy-scipy
import sys
import numpy as np
import scipy.interpolate as si


def bspline(cv, n=100, degree=3, periodic=False):
    """ Calculate n samples on a bspline

        cv :      Array ov control vertices
        n  :      Number of samples to return
        degree:   Curve degree
        periodic: True - Curve is closed
                  False - Curve is open
    """

    # If periodic, extend the point array by count+degree+1
    cv = np.asarray(cv)
    count = len(cv)

    if periodic:
        factor, fraction = divmod(count+degree+1, count)
        cv = np.concatenate((cv,) * factor + (cv[:fraction],))
        count = len(cv)
        degree = np.clip(degree,1,degree)

    # If opened, prevent degree from exceeding count-1
    else:
        degree = np.clip(degree,1,count-1)


    # Calculate knot vector
    kv = None
    if periodic:
        kv = np.arange(0-degree,count+degree+degree-1,dtype='int')
    else:
        kv = np.array([0]*degree + range(count-degree+1) + [count-degree]*degree,dtype='int')

    # Calculate query range
    u = np.linspace(periodic,(count-degree),n)
    u = np.array([1.,2.,3.,4.,5.,6.])
    print u
    # Calculate result
    arange = np.arange(len(u))
    points = np.zeros((len(u),cv.shape[1]))
    for i in xrange(cv.shape[1]):
        points[arange,i] = si.splev(u, (kv,cv[:,i],degree))
    return points

def main():
    cv = np.array([[ 50.,  25., 0.],
                   [ 59.,  12., 0.],
                   [ 50.,  10., 0.],
                   [ 57.,   2., 10.],
                   [ 40.,   4., 0.],
                   [ 40.,   14., 0.]])

    p = bspline(cv,n=100,degree=3,periodic=True)
    x,y,z = p.T
    print x
    return True

if __name__ == '__main__':
    sys.exit(main())

