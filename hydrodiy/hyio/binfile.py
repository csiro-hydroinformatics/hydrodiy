import re
import struct
import datetime
import numpy as np
import pandas as pd

from hydata import dutils

long_types = [np.int64, np.int32, int]

double_types = [np.float64, np.float32, float]

datetime_types = [np.datetime64, np.dtype('<M8[ns]'),
        np.dtype('>M8[ns]')]

def _todatatypes(data, strlength):
    """ Convert data frame according to datatypes """

    data2 = data.values

    dtt = data.dtypes.values

    ncol = data.shape[1]

    datatypes = np.array(['s'] * ncol)

    for i in range(ncol):

        if dtt[i] in double_types:
            datatypes[i] = 'd'
            data2[:, i] = data.iloc[:, i].astype(np.float64)

        elif dtt[i] in long_types:
            datatypes[i] = 'l'
            data2[:, i] = data.iloc[:, i].astype(np.uint64)

        elif dtt[i] in datetime_types:
            datatypes[i] = 't'
            data2[:, i] = data.iloc[:, i].apply(dutils.osec2time).astype(np.int64)

        else:
            datatypes[i] = 's'
            tmp = data.iloc[:, i].astype(str)

            # Pad string with spaces
            tmp = tmp.apply(lambda x: (x+ '\0'*strlength)[:strlength])
            data2[:, i] = tmp.apply(lambda x: 
                        x.encode('ascii', 'ignore')[:strlength])

    return data2, datatypes
   

def _binhead(datatypes, strlength, timestep, nrow, ncol,comment):
    """ Produces the bin files header """

    comment_list = comment
    if not isinstance(comment, list):
        comment_list = [comment]

    h = []
    h.append('%10s %d\n'%('strlength', strlength))
    h.append('%10s %d\n'%('timestep', timestep))
    h.append('%10s %d\n'%('ncol', ncol))
    h.append('%10s %d\n'%('nrow', nrow))
    h.append('%10s %s\n'%('datatypes', ''.join(datatypes)))
    h.append('%10s %s\n'%('comment', comment))

    return h

def write_bin(data, filebase, comment, strlength=30, timestep=-1):
    """ write a pandas dataframe to a bin file with comments """
   
    # Convert to datatypes
    data2, datatypes = _todatatypes(data, strlength)

    # write header 
    head = _binhead(datatypes, strlength, timestep, 
        data.shape[0], data.shape[1], comment) 

    fheader = open('%sh'%filebase, 'w')
    fheader.writelines(head)
    fheader.close()

    # write double data 
    nd = np.sum(datatypes == 'd')
    if nd >0:
        datad = data2[:, datatypes == 'd']
        datad = datad.flat[:]
        databin = struct.pack('d'*len(datad), *datad);

        fbin = open('%sd' % filebase, 'wb')
        fbin.write(databin)
        fbin.close()
        
    # write long int data 
    idx = np.array([(dt == 'l') | (dt == 't') for dt in datatypes])
    nl = np.sum(idx)
    if nl >0:
        datal = data2[:, idx]
        datal = datal.flat[:]
        databin = struct.pack('Q'*len(datal), *datal);

        fbin = open('%sl' % filebase, 'wb')
        fbin.write(databin)
        fbin.close()

    # write string data 
    ns = np.sum(datatypes == 's')
    if ns >0:
        datas = data2[:, datatypes == 's']
        datas = datas.reshape((np.prod(datas.shape),))
        datas = np.array([[c for c in v] for v in datas]).flat[:]
        databin = struct.pack('s'*len(datas), *datas);

        fbin = open('%ss' % filebase, 'wb')
        fbin.write(databin)
        fbin.close()


def read_bin(filebase):
    """ Reads data from bin file with header file to a pandas data frame"""

    # Reads header
    fhead = open('%sh'%filebase, 'r')
    strlength = int(fhead.readline()[10:])
    timestep = int(fhead.readline()[10:])
    ncol = int(fhead.readline()[10:])
    nrow = int(fhead.readline()[10:])
    dt = re.sub(' +', '', fhead.readline()[10:].strip())
    comment = fhead.readline()[10:].strip()
    fhead.close()

    # reshape datatypes
    datatypes = np.array([''] * ncol)
    for i in range(len(dt)):
        datatypes[i] = dt[i]
   
    if len(dt) < len(datatypes):
        datatypes[len(dt):] = datatypes[len(dt)]

    # reads data
    data_all = {}
    config_unpack = {'d':{'flag':'d', 'nbyte':8}, 
        'l':{'flag':'Q', 'nbyte':8}, 
        's':{'flag':'s', 'nbyte':1},
        't':{'flag':'Q', 'nbyte':8}}

    for datatype in ['d', 'l', 's', 't']:

        datam = None
        ncolm = np.sum(datatypes == datatype)

        cfg = config_unpack[datatype]

        if ncolm > 0:
            fbin = open('%s%c' % (filebase, datatype), 'rb')

            datam = []

            i = 0
            chunk = fbin.read(cfg['nbyte'])

            while chunk:
                datam.append(struct.unpack(cfg['flag'], chunk)[0])
                i += 1
                chunk = fbin.read(cfg['nbyte'])

            fbin.close()

            # Reformat string data
            if datatype == 's':
                datam = np.array(datam).reshape(nrow*ncolm, strlength)
                datam = np.array([''.join(v) for v in datam])

            # Convert time data
            if datatype == 't':
                datam = np.array([dutils.osec2time(dt) for dt in datam])
                
            datam = np.array(datam).reshape(nrow, ncolm)

            data_all[datatype] = datam

    # Create final object
    data = {}

    for i in range(ncol):

        if datatypes[i] == 'd':
            idxd = np.sum(datatypes[:i] == 'd')
            data['C%3.3d' % i] = data_all['d'][:, idxd]
        
        if datatypes[i] == 'l':
            idxl = np.sum(datatypes[:i] == 'l')
            data['C%3.3d' % i] = data_all['l'][:, idxl]

        if datatypes[i] == 's':
            idxs = np.sum(datatypes[:i] == 's')
            data['C%3.3d' % i] = data_all['s'][:, idxs]


    data = pd.DataFrame(data)

    return data, strlength, timestep, comment
  
