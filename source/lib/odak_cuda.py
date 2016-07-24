#!/usr/bin/python
# -*- coding: utf-8 -*-

# Whole library can be found under https://github.com/kunguz/odak.
import sys,os,time
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

__author__  = ('Kaan Akşit')

mod = SourceModule("""
  __global__ void doublify(float *a)
  {
    int idx = threadIdx.x + threadIdx.y*4;
    a[idx] *= 2;
  }
  """)

# Definition to transfer numpy arrays to GPU's memory.
def TranToGPU(a):
    a     = a.astype(np.float32)
    a_gpu = cuda.mem_alloc(a.nbytes)
    cuda.memcpy_htod(a_gpu,a)
    return a_gpu

# Definition to apply a function to a numpy array on GPU's memory.
def ApplyGPUFunc(a_gpu,mod,FuncName,blockno=(4,4,1)):
    func  = mod.get_function(FuncName)
    func(a_gpu, block=blockno)
    return a_gpu

# Definition to transfer numpy arrays from GPU's memory.
def TranFromGPU(a_gpu,a):
    a_res = np.empty_like(a.astype(np.float32))
    cuda.memcpy_dtoh(a_res,a_gpu)
    return a_res

# Main definition.
def main():
    a = np.ones((4,4))
    for i in xrange(0,5):
        a_gpu_0 = TranToGPU(a)
        a_gpu_1 = ApplyGPUFunc(a_gpu_0,mod,"doublify",blockno=(4,4,1))
        a_res   = TranFromGPU(a_gpu_1,a)
        print a_res
    print 'Odak by %s' % __author__
    return True

if __name__ == '__main__':
    sys.exit(main())

