
import numpy as np

def foo():

    for i in range(2):

        a = np.ones((4, )) * i

        import os
        from hyio import minilog

        source_file = os.path.abspath(__file__)

        minilog.log({'a':','.join(a.astype(str)), 'i':i}, 'log1_i%d' % i, 
                        source_file, 12)
        minilog.log({'a np':a, 'i':i}, 'log2_i%d' % i, 
                        source_file, 14)
