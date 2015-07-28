
import numpy as np
import pandas as pd

# All code converted from 
# https://source.ggy.bris.ac.uk/wiki/Baseflow_separation

def baseflow(q, params, method=1):

    # Initialisation
    nval = q.shape[0]
    bf = np.ones(nval) * q[0]
    qq = q[0]
    qqp = qq

    # Parameters
    k = min(1., max(0., params[0]))
    if method>=2:
        C = max(0., params[1])
    if method>=3:
        a = min(0., max(0., params[2]))

    # Loop
    for i in range(1, nval):
        
        # Takes care of missing values
        if pd.notnull(q[i]) & (q[i]>=0):
            qq = q[i]

        # Apply base flow method
        if method == 1:
            bf[i] = k*bf[i-1]/(2-k) + (1-k)*qq/(2-k)

        elif method == 2:
            bf[i] = k*bf[i-1]/(1+C) + C*qq/(1+C)

        elif method == 3:
            bf[i] = k*bf[i-1]/(1+C) + C*(qq+a*qqp)/(1+C)

        # Discard floods
        if bf[i]>qq:
            bf[i] = qq

        qqp = qq

    # Compute BFI
    idx = pd.notnull(q) & (q>=0)
    BFI = np.sum(bf[idx]) / np.sum(q[idx])

    return bf, BFI
