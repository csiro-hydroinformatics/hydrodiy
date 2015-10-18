import numpy as np
cimport numpy as np

np.import_array()

# -- HEADERS --
cdef extern from 'c_utils.h':
    int c_utils_getesize(int * esize)

cdef extern from 'c_uh.h':
    int c_uh_getnuhmaxlength()

cdef extern from 'c_uh.h':
    double c_uh_getuheps()

cdef extern from 'c_uh.h':
    int c_uh_getuh(int nuhlengthmax,
            int uhid, 
            double lag,
            int * nuh, 
            double * uh)

cdef extern from 'c_dummy.h':
    int c_dummy_getnstates()

cdef extern from 'c_dummy.h':
    int c_dummy_getnoutputs()

cdef extern from 'c_dummy.h':
    int c_dummy_run(int nval, 
            int nparams, 
            int ninputs,
            int nstates, int noutputs,
    	    double * params,
    	    double * inputs,
    	    double * statesini,
            double * outputs)

def __cinit__(self):
    pass

def getesize(np.ndarray[int, ndim=1, mode='c'] esize not None):

    cdef int ierr

    ierr = c_utils_getesize(<int*> np.PyArray_DATA(esize))

    return ierr


def uh_getnuhmaxlength():
    return c_uh_getnuhmaxlength()

def uh_getuheps():
    return c_uh_getuheps()

def uh_getuh(int nuhlengthmax, int uhid, double lag,
        np.ndarray[int, ndim=1, mode='c'] nuh not None,
        np.ndarray[double, ndim=1, mode='c'] uh not None):

    cdef int ierr

    ierr = c_uh_getuh(nuhlengthmax,
            uhid, 
            lag,
            <int*> np.PyArray_DATA(nuh),
            <double*> np.PyArray_DATA(uh))

    return ierr


def dummy_getnstates():
    return c_dummy_getnstates()

def dummy_getnoutputs():
    return c_dummy_getnoutputs()

def dummy_run(np.ndarray[double, ndim=1, mode='c'] params not None,
        np.ndarray[double, ndim=2, mode='c'] inputs not None,
        np.ndarray[double, ndim=1, mode='c'] statesini not None,
        np.ndarray[double, ndim=2, mode='c'] outputs not None):

    cdef int ierr

    # check dimensions
    if params.shape[0] < 1:
        raise ValueError('params.shape[0] < 1')

    if statesini.shape[0] < 1:
        raise ValueError('statesini.shape[0] < 1')

    if inputs.shape[0] != outputs.shape[0]:
        raise ValueError('inputs.shape[0] != outputs.shape[0]')

    if inputs.shape[1] != 2:
        raise ValueError('inputs.shape[1] != 2')

    ierr = c_dummy_run(inputs.shape[0], 
            params.shape[0], \
            inputs.shape[1], 
            statesini.shape[0], 
            outputs.shape[1], \
            <double*> np.PyArray_DATA(params), \
            <double*> np.PyArray_DATA(inputs), \
            <double*> np.PyArray_DATA(statesini), \
            <double*> np.PyArray_DATA(outputs))

    return ierr

