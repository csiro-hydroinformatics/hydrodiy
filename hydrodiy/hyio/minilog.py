import os, json, tempfile

import datetime

now = datetime.datetime.now

# Default log file
logpath = os.path.join(tempfile.gettempdir(), 
        'log_%s.log' % now().strftime('%Y-%m-%d_%H-%m-%S'))


def log(logdata, label, source, level='DEBUG'):
    ''' Log data to the log file '''

    # Convert logdata to flat list
    if isinstance(logdata, str):
        logdata = [logdata]

    # create json object
    js = {
            'time': now().strftime('%Y-%m-%d_%H-%M-%S'),
            'label':label,
            'source':source,
            'level':level,
        }

    for im in range(len(logdata)):
        js['logdata_%3.3d' % (im+1)] = logdata[im]

    # Write dict data to file
    fs = open(logpath, 'a')

    json.dump(js, fs, indent=2, sort_keys=True)
    fs.write('\n,\n')

    fs.close()


def load():
    ''' Load log file '''

    fs = open(logpath, 'r')
    logtxt = fs.readlines()
    fs.close()

    logtxt = ['{"logdata":['] + logtxt[:-1] + [']}']
    log = json.loads(''.join(logtxt))['logdata']

    return log
    
