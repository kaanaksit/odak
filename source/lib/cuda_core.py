#!/usr/bin/python
# -*- coding: utf-8 -*-

__author__  = ('Kaan Akşit')

kernel = """

         __global__ void subtract_vector(float *p1, float *p2, float *distance)
        {
            const int i = threadIdx.x;
            distance[i] = p1[i] - p2[i];
         }

         __global__ void second_power(float *p1, float *result)
        {
            const int i = threadIdx.x;
            result[i]   = p1[i]*p1[i];
         }

         """

