#!/usr/bin/env python

import os
import re

# Retrieve list of packages
with file('setup.py','r') as f:
    setup_file = '\n'.join(f.readlines())
l = re.findall(r'packages=\[[^\]]*\]',setup_file)
s =  re.findall(r'\'\w+', l[0])
module_names = [re.sub(r'\'', '', u) for u in s]

l = re.findall(r'\(name=.*', setup_file)
package_name = re.sub(r'.*\(name=\'|\',', '', l[0])

print('\n\n----- uninstall info -----')
print('package : %s'%package_name)
print('modules : %s\n\n'%(' '.join(module_names)))

pattern = '|'.join(module_names)
pattern = '|'.join([pattern, package_name])

# Try to install the package and list files
os.system('python setup.py install --user --record package_files.txt')
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

