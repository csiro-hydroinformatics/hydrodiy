import numpy as np
cimport numpy as np

np.import_array()

cdef extern from 'c_match.h':
    int c_match(int nval1, int nval2, 
        double * x1, double * y1, 
        double * x2, double * y2, int * match_final)

def match(np.ndarray[double, ndim=1, mode='c'] x1 not None,
            np.ndarray[double, ndim=1, mode='c'] y1 not None,
            np.ndarray[double, ndim=1, mode='c'] x2 not None,
            np.ndarray[double, ndim=1, mode='c'] y2 not None,
            np.ndarray[int, ndim=1, mode='c'] match_final not None):
    
    assert x1.shape[0]==y1.shape[0]
    assert x2.shape[0]==y2.shape[0]
    assert x2.shape[0]<=x1.shape[0]
    cdef int ierr
    ierr = c_match(x1.shape[0], x2.shape[0], 
            <double*> np.PyArray_DATA(x1),
            <double*> np.PyArray_DATA(y1),
            <double*> np.PyArray_DATA(x2),
            <double*> np.PyArray_DATA(y2),
            <int*> np.PyArray_DATA(match_final))
    return ierr

