import numpy as np
cimport numpy as np

np.import_array()

cdef extern from 'c_armodels.h':
    int c_armodel_sim(int nval, int nparams,
            double sim_mean,
            double sim_ini, double * params,
            double * innov, double* outputs);

    int c_armodel_residual(int nval, int nparams,
            double sim_mean,
            double sim_ini, double * params,
            double * inputs, double* residuals);


cdef extern from 'c_crps.h':
    int c_crps(int nval,int ncol,
        int use_weights, int is_sorted,
        double* obs,
        double* sim,
        double* weights_vector,
        double* reliability_table,
        double* crps_decompos)

cdef extern from 'c_olsleverage.h':
    int c_olsleverage(int nval, int npreds, double * predictors,
        double * tXXinv, double* leverages)


cdef extern from 'c_dscore.h':
    int c_ensrank(double eps, int nval, int ncol, double* sim,
        double * fmat, double * ranks)

cdef extern from 'c_andersondarling.h':
    int c_ad_test(int nval, double *unifdata, double *outputs)


def __cinit__(self):
    pass


def olsleverage(np.ndarray[double, ndim=2, mode='c'] predictors not None,
        np.ndarray[double, ndim=2, mode='c'] tXXinv not None,
        np.ndarray[double, ndim=1, mode='c'] leverages not None):

    cdef int ierr

    # check dimensions
    assert predictors.shape[0] == leverages.shape[0]
    assert predictors.shape[1] == tXXinv.shape[0]
    assert tXXinv.shape[1] == tXXinv.shape[0]

    ierr = c_olsleverage(predictors.shape[0],
            predictors.shape[1],
            <double*> np.PyArray_DATA(predictors),
            <double*> np.PyArray_DATA(tXXinv),
            <double*> np.PyArray_DATA(leverages))

    return ierr


def armodel_sim(double sim_mean, double  sim_ini,
        np.ndarray[double, ndim=1, mode='c'] params not None,
        np.ndarray[double, ndim=1, mode='c'] inputs not None,
        np.ndarray[double, ndim=1, mode='c'] outputs not None):

    cdef int ierr

    # check dimensions
    assert inputs.shape[0] == outputs.shape[0]

    ierr = c_armodel_sim(inputs.shape[0],
            params.shape[0], sim_mean, sim_ini,
            <double*> np.PyArray_DATA(params),
            <double*> np.PyArray_DATA(inputs),
            <double*> np.PyArray_DATA(outputs))

    return ierr


def armodel_residual(double sim_mean, double sim_ini,
        np.ndarray[double, ndim=1, mode='c'] params not None,
        np.ndarray[double, ndim=1, mode='c'] inputs not None,
        np.ndarray[double, ndim=1, mode='c'] residuals not None):

    cdef int ierr

    # check dimensions
    assert inputs.shape[0] == residuals.shape[0]

    ierr = c_armodel_residual(inputs.shape[0],
            params.shape[0], sim_mean, sim_ini,
            <double*> np.PyArray_DATA(params),
            <double*> np.PyArray_DATA(inputs),
            <double*> np.PyArray_DATA(residuals))

    return ierr


def crps(int use_weights, int is_sorted,
        np.ndarray[double, ndim=1, mode='c'] obs not None,
        np.ndarray[double, ndim=2, mode='c'] sim not None,
        np.ndarray[double, ndim=1, mode='c'] weight_vector not None,
        np.ndarray[double, ndim=2, mode='c'] reliability_table not None,
        np.ndarray[double, ndim=1, mode='c'] crps_decompos not None):

    cdef int ierr

    # check dimensions
    assert obs.shape[0]==sim.shape[0]
    assert crps_decompos.shape[0]==5
    assert reliability_table.shape[0]==sim.shape[1]+1
    assert reliability_table.shape[1]==7

    ierr = c_crps(sim.shape[0], sim.shape[1],
            use_weights, is_sorted,
            <double*> np.PyArray_DATA(obs),
            <double*> np.PyArray_DATA(sim),
            <double*> np.PyArray_DATA(weight_vector),
            <double*> np.PyArray_DATA(reliability_table),
            <double*> np.PyArray_DATA(crps_decompos))

    return ierr


def ensrank(double eps,
        np.ndarray[double, ndim=2, mode='c'] sim not None,
        np.ndarray[double, ndim=2, mode='c'] fmat not None,
        np.ndarray[double, ndim=1, mode='c'] ranks not None):

    cdef int ierr

    # check dimensions
    assert sim.shape[0]==ranks.shape[0]
    assert sim.shape[0]==fmat.shape[0]
    assert sim.shape[0]==fmat.shape[1]

    ierr = c_ensrank(eps, sim.shape[0], sim.shape[1],
            <double*> np.PyArray_DATA(sim),
            <double*> np.PyArray_DATA(fmat),
            <double*> np.PyArray_DATA(ranks))

    return ierr


def ad_test(np.ndarray[double, ndim=1, mode='c'] unifdata not None,
        np.ndarray[double, ndim=1, mode='c'] outputs not None):

    # Check dimensions
    assert outputs.shape[0] == 2

    return c_ad_test(unifdata.shape[0],
            <double*> np.PyArray_DATA(unifdata),
            <double*> np.PyArray_DATA(outputs))


