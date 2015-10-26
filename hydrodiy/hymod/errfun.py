
import numpy as np


def sse(obs, sim):
    err = obs-sim
    return np.sum(err*err)

def ssqe_bias(obs, sim):
    err = np.sqrt(obs)-np.sqrt(sim)
    E = np.sum(err*err)
    B = np.mean(obs-sim)
    return E*(1+abs(B))
    
    
