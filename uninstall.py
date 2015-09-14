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
fso = []

for fn in files:
    dd = os.path.dirname(fn)
    if re.search(pattern, dd) is not None:
        dirn.append(dd)

    if re.search('egg-info', fn) is not None:
        fegg = fn.strip()

    if re.search('.*\.so', fn) is not None:
        fso.append(fn.strip())

dirn = set(dirn)
fso = set(fso)

# remove packages folders
for d in dirn:
    print('removing folder %s'%d)
    os.system('rm -r %s'%d)

# remove so lib
for f in fso:
    print('removing lib %s' % f)
    if not os.path.exists(f):
        raise ValueError('%f does not exist, cannot remove' % f)
    os.system('rm -r %s' % f)


# remove egg file
print('removing %s'%fegg)
if not os.path.exists(fegg):
    raise ValueError('%f does not exist, cannot remove' % f)
os.system('rm %s' % fegg)

# remove egg folder
degg = '%s.egg-info' % package_name
print('removing %s' % degg)
os.system('rm -r %s' % degg)

# Remove build and dist folders
print('removing build and dist')
os.system('rm -r build')
os.system('rm -r dist')

