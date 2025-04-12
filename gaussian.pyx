# coding=utf-8
# Copyright (c) DIRECT Contributors
# using the code from 
# https://github.com/NKI-AI/direct/blob/main/direct/common/_gaussian.pyx 
#cython: cdivision=True
#cython: boundscheck=False
#cython: nonecheck=False
#cython: wraparound=False
#cython: overflowcheck=False
#cython: unraisable_tracebacks=False

import numpy as np

cimport numpy as cnp
from libc.math cimport cos, log, pi, sin, sqrt
from libc.stdlib cimport RAND_MAX, rand, srand

cnp.import_array()


cdef double random_uniform() nogil:
    """Produces a random number in (0, 1)."""
    cdef double r = rand()
    return r / RAND_MAX


cdef cnp.ndarray[cnp.float_t, ndim=1, mode='c'] random_normal_1d(
    double mu, double std
):
    """Produces a random vector from the Gaussian distribution based on the Box-Muller algorithm."""
    cdef double r, theta, x

    r = sqrt(-2 * log(random_uniform()))
    theta = 2 * pi * random_uniform()

    x = mu + r * cos(theta) * std

    return np.array([x], dtype=float)



def gaussian_mask_1d(
    int nonzero_count,
    int n,
    int center,
    double std,
    cnp.ndarray[cnp.float_t, ndim=1, mode='c'] mask,
    int seed,
):
    cdef int count, ind
    cdef cnp.ndarray[cnp.float_t, ndim=1, mode='c'] rnd_normal

    srand(seed)

    count = 0

    while count <= nonzero_count:
        rnd_normal = random_normal_1d(center, std)
        ind = int(rnd_normal[0])
     
        if 0 <= ind < n and mask[ind] != 1:
            mask[ind] = 1
            count = count + 1