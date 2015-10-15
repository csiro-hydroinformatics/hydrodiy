import numpy as np
cimport numpy as np

np.import_array()

# To force initialisation - Cython compilation bugs otherwise
init = ""

# -- HEADERS --

cdef extern from 'c_utils.h':
    int c_utils_getesize()

cdef extern from 'c_uh.h':
    int c_uh_getnuhmaxlength()

cdef extern from 'c_uh.h':
    double c_uh_getuheps()

cdef extern from 'c_uh.h':
    int c_uh_getuh(int uhid, double lag,
            int * nuh, double * uh)

cdef extern from 'c_gr4j.h':
    int c_gr4j_getnstates()

cdef extern from 'c_gr4j.h':
    int c_gr4j_getnoutputs()

cdef extern from 'c_gr4j.h':
    int c_gr4j_run(int nval, int nparams, int nuh1, int nuh2,
            int ninputs,
            int nstates, int noutputs,
    	    double * params,
            double * uh1,
            double * uh2,
    	    double * inputs,
            double * statesuhini,
    	    double * statesini,
            double * outputs)

cdef extern from 'c_gr2m.h':
    int c_gr2m_getnstates()

cdef extern from 'c_gr2m.h':
    int c_gr2m_getnoutputs()

cdef extern from 'c_gr2m.h':
    int c_gr2m_getesize()

cdef extern from 'c_gr2m.h':
    int c_gr2m_run(int nval, int nparams, int ninputs,
            int nstates, int noutputs,
    	    double * params,
    	    double * inputs,
    	    double * statesini,
            double * outputs)


def getesize():
    return c_utils_getesize()

# -- UH FUNCTIONS --

def uh_getnuhmaxlength():
    return c_uh_getnuhmaxlength()

def uh_getuheps():
    return c_uh_getuheps()

def uh_getuh(int uhid, double lag,
        np.ndarray[int, ndim=1, mode='c'] nuh not None,
        np.ndarray[double, ndim=1, mode='c'] uh not None):

    cdef int ierr

    ierr = c_uh_getuh(uhid, lag,
            <int*> np.PyArray_DATA(nuh),
            <double*> np.PyArray_DATA(uh))

    return ierr


# -- GR4J FUNCTIONS --

def gr4j_getnstates():
    return c_gr4j_getnstates()


def gr4j_getnoutputs():
    return c_gr4j_getnoutputs()

def gr4j_run(int nuh1, int nuh2,
        np.ndarray[double, ndim=1, mode='c'] params not None,
        np.ndarray[double, ndim=1, mode='c'] uh1 not None,
        np.ndarray[double, ndim=1, mode='c'] uh2 not None,
        np.ndarray[double, ndim=2, mode='c'] inputs not None,
        np.ndarray[double, ndim=1, mode='c'] statesuhini not None,
        np.ndarray[double, ndim=1, mode='c'] statesini not None,
        np.ndarray[double, ndim=2, mode='c'] outputs not None):

    cdef int ierr

    # check dimensions
    assert params.shape[0] == 4
    assert statesini.shape[0] >= 2
    assert inputs.shape[0] == outputs.shape[0]
    assert inputs.shape[1] == 2
    assert uh1.shape[0] >= nuh1
    assert uh2.shape[0] >= nuh2

    ierr = c_gr4j_run(inputs.shape[0], params.shape[0], \
            nuh1, nuh2, \
            2, 2, outputs.shape[1], \
            <double*> np.PyArray_DATA(params), \
            <double*> np.PyArray_DATA(uh1), \
            <double*> np.PyArray_DATA(uh2), \
            <double*> np.PyArray_DATA(inputs), \
            <double*> np.PyArray_DATA(statesuhini), \
            <double*> np.PyArray_DATA(statesini), \
            <double*> np.PyArray_DATA(outputs))

    return ierr


# -- GR2M FUNCTIONS --

def gr2m_getnstates():
    return c_gr2m_getnstates()

def gr2m_getnoutputs():
    return c_gr2m_getnoutputs()

def gr2m_run(np.ndarray[double, ndim=1, mode='c'] params not None,
        np.ndarray[double, ndim=2, mode='c'] inputs not None,
        np.ndarray[double, ndim=1, mode='c'] statesini not None,
        np.ndarray[double, ndim=2, mode='c'] outputs not None):

    cdef int ierr

    # check dimensions
    assert params.shape[0] == 2
    assert statesini.shape[0] >= 2
    assert inputs.shape[0] == outputs.shape[0]
    assert inputs.shape[1] == 2

    ierr = c_gr2m_run(inputs.shape[0], params.shape[0], \
            2, 2, outputs.shape[1], \
            <double*> np.PyArray_DATA(params), \
            <double*> np.PyArray_DATA(inputs), \
            <double*> np.PyArray_DATA(statesini), \
            <double*> np.PyArray_DATA(outputs))

    return ierr


