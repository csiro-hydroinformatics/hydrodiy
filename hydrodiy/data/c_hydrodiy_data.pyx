import numpy as np
cimport numpy as np

np.import_array()

cdef extern from 'c_dateutils.h':
    int c_dateutils_isleapyear(int year)
    int c_dateutils_daysinmonth(int year, int month)
    int c_dateutils_dayofyear(int month, int day)
    int c_dateutils_add1month(int * date)
    int c_dateutils_add1day(int * date)
    int c_dateutils_comparedates(int * date1, int * date2)
    int c_dateutils_getdate(double day, int * date)

cdef extern from 'c_dutils.h':
    int c_aggregate(int nval, int oper, int * aggindex,
            double * inputs, double * outputs, int * iend);

def __cinit__(self):
    pass


def isleapyear(int year):
    return c_dateutils_isleapyear(year)


def daysinmonth(int year, int month):
    return c_dateutils_daysinmonth(year, month)


def dayofyear(int month, int day):
    return c_dateutils_dayofyear(month, day)


def add1month(np.ndarray[int, ndim=1, mode='c'] date not None):

    cdef int ierr

    # check dimensions
    assert date.shape[0] == 3

    ierr = c_dateutils_add1month(<int*> np.PyArray_DATA(date))

    return ierr


def add1day(np.ndarray[int, ndim=1, mode='c'] date not None):

    cdef int ierr

    # check dimensions
    assert date.shape[0] == 3

    ierr = c_dateutils_add1day(<int*> np.PyArray_DATA(date))

    return ierr


def comparedates(np.ndarray[int, ndim=1, mode='c'] date1 not None,
        np.ndarray[int, ndim=1, mode='c'] date2 not None):

    cdef int ierr

    # check dimensions
    assert date1.shape[0] == 3
    assert date2.shape[0] == 3

    ierr = c_dateutils_comparedates(
        <int*> np.PyArray_DATA(date1),
        <int*> np.PyArray_DATA(date2))

    return ierr


def getdate(double day,
    np.ndarray[int, ndim=1, mode='c'] date not None):

    cdef int ierr

    # check dimensions
    assert date.shape[0] == 3

    ierr = c_dateutils_getdate(day, <int*> np.PyArray_DATA(date))

    return ierr


def aggregate(int oper,
        np.ndarray[int, ndim=1, mode='c'] aggindex not None,
        np.ndarray[double, ndim=1, mode='c'] inputs not None,
        np.ndarray[double, ndim=1, mode='c'] outputs not None,
        np.ndarray[int, ndim=1, mode='c'] iend not None):

    cdef int ierr

    # check dimensions
    assert aggindex.shape[0] == inputs.shape[0]
    assert aggindex.shape[0] == outputs.shape[0]
    assert iend.shape[0] == 1

    ierr = c_aggregate(inputs.shape[0], oper,
            <int*> np.PyArray_DATA(aggindex),
            <double*> np.PyArray_DATA(inputs),
            <double*> np.PyArray_DATA(outputs),
            <int*> np.PyArray_DATA(iend))

    return ierr


def aggregate(int oper,
        np.ndarray[int, ndim=1, mode='c'] aggindex not None,
        np.ndarray[double, ndim=1, mode='c'] inputs not None,
        np.ndarray[double, ndim=1, mode='c'] outputs not None,
        np.ndarray[int, ndim=1, mode='c'] iend not None):

    cdef int ierr

    # check dimensions
    assert aggindex.shape[0] == inputs.shape[0]
    assert aggindex.shape[0] == outputs.shape[0]
    assert iend.shape[0] == 1

    ierr = c_aggregate(inputs.shape[0], oper,
            <int*> np.PyArray_DATA(aggindex),
            <double*> np.PyArray_DATA(inputs),
            <double*> np.PyArray_DATA(outputs),
            <int*> np.PyArray_DATA(iend))

    return ierr


