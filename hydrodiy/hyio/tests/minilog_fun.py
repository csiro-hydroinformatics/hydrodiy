
import numpy as np

def foo():

    for i in range(2):
        a = np.ones((4, )) * i

        import os
        from hyio import minilog
        source_file = os.path.abspath(__file__)
        minilog.log(','.join(a.astype(str)), 'a_vect', source_file)
        
