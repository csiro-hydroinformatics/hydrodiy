#!/usr/bin/env python

import os, re, json
import numpy as np

js = 'package_config.json'
package_config = json.load(open(js, 'r'))

package_name = package_config['name']
module_names = package_config['packages']


print('\n\n----- uninstall info -----')
print('package : %s'%package_name)
print('modules : %s\n\n'%(' '.join(module_names)))

pattern = '|'.join(module_names)
pattern = '|'.join([pattern, package_name])

# Try to install the package and list files
# add --user if no venv
os.system('python setup.py install --record package_files.txt')
with file('package_files.txt', 'r') as fp:
    files = fp.readlines()


# Retrieve package folder
dirn = []
fegg = None
for fn in files:
    dd = os.path.dirname(fn)
    se = re.search(pattern, dd)
    if se is not None:
        dirn.append(dd)
    se = re.search('egg-info', fn)
    if se is not None:
        fegg = fn
dirn = set(dirn)

# remove packages folders
for d in dirn:
    print('removing folder %s\n'%d)
    os.system('rm -r %s'%d)

# remove egg file
print('removing %s\n'%fegg)
os.system('rm %s'%fegg)

# remove egg folder
degg = '%s.egg-info'%package_name
print('removing %s\n'%degg)
os.system('rm -r %s'%degg)

# Remove build and dist folders
print('removing build and dist')
os.system('rm -r build')
os.system('rm -r dist')

