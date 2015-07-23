import os, json, tempfile

import datetime

now = datetime.datetime.now

# Default log file
logpath = os.path.join(tempfile.gettempdir(), 
        'log_%s.log' % now().strftime('%Y-%m-%d_%H-%m-%S'))


def log(logdata, label, source, line, level='DEBUG'):
    ''' Log data to the log file '''

    # Convert logdata to dict
    if isinstance(logdata, str):
        logdata = {'item000':logdata}

    if isinstance(logdata, list):
        lgd = {}
        for i in range(len(logdata)):
            lgd['item%3.3d' % i] = logdata[i]

        logdata = lgd

    # create json object
    js = {
            'time': now().strftime('%Y-%m-%d_%H-%M-%S'),
            'label':label,
            'source':source,
            'line':line,
            'level':level,
            'logdata': logdata
        }

    # Write dict data to file
    fs = open(logpath, 'a')
    
    try:
        jscode = json.dumps(js, indent=4, sort_keys=True)
    except (TypeError, ValueError) as e:
        js['logdata'] = {'error':'%s' % e}
    
    json.dump(js, fs, indent=4, sort_keys=True)

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
    
