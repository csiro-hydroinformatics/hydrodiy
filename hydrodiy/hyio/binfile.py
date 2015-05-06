import time
import struct
import numpy as np
import pandas as pd

def _getdatatypes(data):
    """ Get the data types from a data frame """
    
    dtt = data.dtypes

    dt = np.array(['s'] * len(dtt))

    dt[(dtt == np.int64).values] = 'l'
    dt[(dtt == np.float64).values] = 'd'

    return dt

def _todatatypes(data, datatypes, strlength):
    """ Convert data frame according to datatypes """

    data2 = data.copy()

    for i in range(len(datatypes)):

        dt = datatypes[i]

        if dt == 'd':
            data2.iloc[:, i] = data.iloc[:, i].astype(np.float64)

        if dt == 'l':
            data2.iloc[:, i] = data.iloc[:, i].astype(np.int64)

        if dt == 'd':
            data2.iloc[:, i] = data.iloc[:, i].astype(str).apply(lambda x: 
                        x.encode('ascii', 'ignore')[:strlength])
   
    return data2
    

def _binhead(datatypes, strlength, nrow, ncol,comment):
    """ Produces the bin files header """

    comment_list = comment
    if not isinstance(comment, list):
        comment_list = [comment]

    h = []
    h.append('%10s %d\n'%('strlength', strlength))
    h.append('%10s %d\n'%('ncol', ncol))
    h.append('%10s %d\n'%('nrow', nrow))
    h.append('%10s %s\n'%('datatypes', datatypes))
    h.append('%10s %s\n'%('comment', comment))

    return h

def write_bin(data, filebase, comment, strlength=30):
    """ write a pandas dataframe to a bin file with comments """
   
    # Get datatypes
    datatypes = _getdatatypes(data)

    # Convert to datatypes
    data2 = _todatatypes(data, datatypes, strlength)

    # write header 
    head = _binhead(datatypes, strlength, 
        data.shape[0], data.shape[1], comment) 

    fheader = open('%sh'%filebase, 'w')
    fheader.writelines(head)
    fheader.close()

    # write double data 
    nd = np.sum(datatypes == 'd')
    if nd >0:
        datad = data2.loc[:, datatypes == 'd'].values
        datad = datad.reshape((np.prod(datad.shape),))
        databin = struct.pack('d'*len(datad), *datad);

        fbin = open('%sd' % filebase, 'wb')
        fbin.write(databin)
        fbin.close()

    # write long int data 
    nd = np.sum(datatypes == 'l')
    if nd >0:
        datal = data2.loc[:, datatypes == 'l'].values
        datal = datal.reshape((np.prod(datal.shape),))
        databin = struct.pack('Q'*len(datal), *datal);

        fbin = open('%sl' % filebase, 'wb')
        fbin.write(databin)
        fbin.close()


def read_bin(filebase):
    """ Reads data from bin file with header file to a pandas data frame"""

    # Reads header
    fhead = open('%sh'%filebase, 'r')
    strlength = int(fhead.readline()[10:])
    ncol = int(fhead.readline()[10:])
    nrow = int(fhead.readline()[10:])
    dt = fhead.readline()[10:]
    comment = fhead.readline()[10:]
    fhead.close()

    # reshape datatypes
    datatypes = np.array([''] * ncol)
    for i in range(len(dt)):
        datatypes[i] = dt[i]
   
    if len(dt) < len(datatypes):
        datatypes[len(dt):] = datatypes[len(dt)]

    # reads double data
    datad = None
    ncold = np.sum(datatypes == 'd')
    if nd > 0:
        fbin = open('%sd' % filebase, 'rb')
        datad = np.zeros((ncold*nrow,))
        i = 0
        chunk = fbin.read(nbytes)
        while chunk:
            datad[i] = struct.unpack('d', chunk)[0]
            i += 1
            chunk = fbin.read(nbytes)
        fbin.close()

        datad = pd.DataFrame(datad.reshape(nrow, ncold))

    # reads long data
    datal = None
    ncoll = np.sum(datatypes == 'l')
    if nd > 0:
        fbin = open('%sl' % filebase, 'rb')
        datal = np.zeros((ncoll*nrow,))
        i = 0
        chunk = fbin.read(nbytes)
        while chunk:
            datal[i] = struct.unpack('Q', chunk)[0]
            i += 1
            chunk = fbin.read(nbytes)
        fbin.close()

        datal = pd.DataFrame(datad.reshape(nrow, ncoll))

    data = pd.merge([datad, datal], axis=1) 

    return data, comment

