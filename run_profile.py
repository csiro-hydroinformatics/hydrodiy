#!/usr/bin/env python

import sys
import os
import re
from hyio import iutils

folder = 'hydrodiy'
pattern = 'tests.*profile.*\\.py$'

# find test files
test_files = iutils.find_files(folder, pattern)

# run profiles
i = 1
for fn in test_files:
    print('\n(%2d) running %s:\n'%(i,fn))
    i += 1

    fn_out = re.sub('py$', 'pstats', fn)
    cmd = 'python -m cProfile -o {0} {1}'.format(fn_out, fn)
    os.system(cmd)

    fn_png = re.sub('py$', 'png', fn)
    cmd = ('gprof2dot -f pstats {0} ' + \
            '| dot -Tpng -o {1}').format(fn_out, 
                fn_png)
    os.system(cmd)



