
import numpy as np
import inspect

def foo():

    for i in range(2):

        a = np.ones((4, )) * i

        import os
        from hydrodiy.io import minilog

        line, fname = minilog.where()
        minilog.log({'a':','.join(a.astype(str)), 'i':i}, 'log1_i%d' % i,
                        __file__, line, fname)

        line, fname = minilog.where()
        minilog.log({'a np':a, 'i':i}, 'log2_i%d' % i,
                        __file__, line, fname)
