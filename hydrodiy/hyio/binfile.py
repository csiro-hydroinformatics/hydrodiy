import time
import struct
import numpy as np
import pandas as pd

def _binhead(nbytes, nrow, ncol,comment):
    """ Produces a nice header for bin files """
    comment_list = comment
    if not isinstance(comment, list):
        comment_list = [comment]

    h = []
    h.append('%10s %d\n'%('nbytes', nbytes))
    h.append('%10s %d\n'%('ndim1', ncol))
    h.append('%10s %d\n'%('ndim2', nrow))
    h.append('%10s %s\n'%('comment', comment))

    return h

def write_bin(data, filename, comment):
    """ write a pandas dataframe to a bin file with comments """
   
    # write header 
    head = _binhead(8, data.shape[0], data.shape[1], comment)
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

    return data, comment

