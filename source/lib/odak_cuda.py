#!/usr/bin/python
# -*- coding: utf-8 -*-

# Whole library can be found under https://github.com/kunguz/odak.
import sys,os,time
import numpy as np
import cuda_core
import pycuda.driver as cuda
import pycuda.gpuarray as gpuarray
import pycuda.autoinit
from pycuda.compiler import SourceModule

__author__  = ('Kaan Akşit')

# 3D Ray tracing library ported to CUDA from CPU, original work is in:
# https://github.com/kunguz/odak/blob/master/source/lib/odak.py#L95
class raytracing():
    # Initialize.
    def __init__(self):
        self.kernel  = cuda_core.kernel
        self.cudaray = SourceModule(self.kernel)
        return
    # Definition to transfer numpy arrays to GPU's memory.
    def TranToGPU(self,a):
        a     = a.astype(np.float32)
        a_gpu = gpuarray.to_gpu(a)
        return a_gpu
    # Definition to transfer numpy arrays from GPU's memory.
    def TranFromGPU(self,a_gpu,a):
        a_res = np.empty_like(a.astype(np.float32))
        cuda.memcpy_dtoh(a_res,a_gpu)
        return a_res
    # Definition to apply a function to a numpy array on GPU's memory.
    def finddistancebetweentwopoints(self,p1_gpu,p2_gpu,blockno=(3,1,1)):
        dist     =  (gpuarray.sum(((p1_gpu-p2_gpu)**2))**0.5).get()
        return dist
    # Definition to find angle between two points if there was a line intersecting at both of them.
    def findangles(self,p1_gpu,p2_gpu):
        return angles
#        # Subtract points from each other.
#        func     = self.cudaray.get_function("subtract_vector")
#        dist     = np.zeros((3,1))
#        dist_gpu = self.TranToGPU(dist)
#        func(p1_gpu,p2_gpu,dist_gpu,block=blockno)
#        # Element-wise second power.
#        func     = self.cudaray.get_function("second_power")
#        func(dist_gpu,dist_gpu,block=blockno)
#        # Sum all the vector.
#        dist     = gpuarray.sum(dist_gpu)
#        return dist

# Main definition.
def main():
    # Define ray tracing envorinment.
    ray      = raytracing()
    # Dummy points in space.
    p1       = np.array([1.,4.,2.])
    p2       = np.array([6.,1.,5.])
    # Move them to GPU.
    p1_gpu   = ray.TranToGPU(p1)
    p2_gpu   = ray.TranToGPU(p2)
    # Calculate the distance between them on GPU.
    dist_gpu = ray.finddistancebetweentwopoints(p1_gpu,p2_gpu)
    print dist_gpu
    # Move it back to the memory from GPU.
    print 'Odak by %s' % __author__
    return True

if __name__ == '__main__':
    sys.exit(main())


