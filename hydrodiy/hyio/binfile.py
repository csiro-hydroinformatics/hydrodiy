import time
import struct
import numpy as np
import pandas as pd

def _binhead(nbytes, nrow, ncol,comment, calendarstart, timestep_duration_sec):
    """ Produces a nice header for bin files """
    comment_list = comment
    if not isinstance(comment, list):
        comment_list = [comment]

    h = []
    h.append('%10s %d\n'%('nbytes', nbytes))
    h.append('%10s %d\n'%('ndim1', ncol))
    h.append('%10s %d\n'%('ndim2', nrow))
    h.append('%10s %s\n'%('comment', comment))
    h.append('%10s %0.0f\n'%('start', calendarstart))
    h.append('%10s %d\n'%('dt_sec', timestep_duration_sec))

    return h

def write_bin(data, filename, comment, calendarstart=-999, timestep_duration_sec=-999):
    """ write a pandas dataframe to a bin file with comments """
   
    # write header 
    head = _binhead(8, data.shape[0], data.shape[1], comment, 
        calendarstart, timestep_duration_sec)
    fheader = open('%sh'%filename, 'w')
    fheader.writelines(head)
    fheader.close()

    # write data 
    datan = data.astype('float64').values
    datan = datan.reshape((np.prod(data.shape),))
    databin = struct.pack('d'*len(datan), *datan);
    fbin = open(filename, 'wb')
    fbin.write(databin)
    fbin.close()

def read_bin(filename):
    """ Reads data from bin file with header file to a pandas data frame"""

    # Reads header
    fhead = open('%sh'%filename, 'r')
    nbytes = int(fhead.readline()[10:])
    ndim1 = int(fhead.readline()[10:])
    ndim2 = int(fhead.readline()[10:])
    comment = fhead.readline()[10:]

    try:
        calendarstart = float(fhead.readline()[10:])
        timestep_duration_sec = int(fhead.readline()[10:])
    except :
        calendarstart = -999.
        timestep_duration_sec = -999

    fhead.close()

    # reads data
    fbin = open(filename, 'rb')
    datan = np.zeros((ndim1*ndim2,))
    i = 0
    chunk = fbin.read(nbytes)
    while chunk:
        if nbytes == 4:
            datan[i] = struct.unpack('f', chunk)[0]
        if nbytes == 8:
            datan[i] = struct.unpack('d', chunk)[0]
        i += 1
        chunk = fbin.read(nbytes)
    fbin.close()
    data = pd.DataFrame(datan.reshape(ndim2, ndim1))

    return data, comment, calendarstart, timestep_duration_sec

