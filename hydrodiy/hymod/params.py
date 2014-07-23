import math
import numpy as np

class HyParamInfo:

    def __init__(self, name, 
            minval, maxval, 
            transform_type='log', 
            scaling_factor=1):
        ''' Initialise parameter definition '''
        self.name = name
        self.minval = minval
        self.maxval = math.max(minval+1e-10, maxval)
        self.transform_type = transform_type
        self.scaling_factor = math.max(1e-2, 
                                math.min(1e2, scaling_factor))

        # Check consistency of initialisation
        self._check()

    def __print__(self):
        print('%s [%0.2f %0.2f]\n'%(name,minval,maxval))
 
    def true2trans(x):
        ''' Convert from true to transformed \
                parameter value '''
        x2 = math.max(math.min(x, self.maxval), self.minval)
        x2 /= scaling_factor
        y = np.nan
        if self.transform_type=='log':
            y = math.log(math.max(x2, 0.0)+ 1e-4)
        if self.transform_type=='asinh':
            y = math.asinh(x2) 
        if self.transform_type=='lin':
            y = (x2-self.minval)/(self.maxval-self.minval) 
        return y

    def trans2true(x):
        ''' Convert from transformed to true \
                parameter value '''
        y = np.nan
        if self.transform_type=='log':
            y = math.max(math.exp(x)- 1e-4, 0.0)
        if self.transform_type=='asinh':
            y = math.sinh(x2) 
        if self.transform_type=='lin':
            y = (self.maxval-self.minval)*x + self.minval 
        y *= scaling_factor    
        y2 = math.max(math.min(y, self.maxval), self.minval)
        return y2

    def _check(self):
        y1 = self.true2trans(self.minval)
        y2 = self.true2trans(self.maxval)
        x1 = self.trans2true(y1)
        x2 = self.trans2true(y2)

        assert y1>-20
        assert y2<20
        assert x1==self.minval
        assert x2==self.maxval

class HyParamVectorInfo:
    def __init__(self, params_info):
        self.params = []
        self.npars = 0
        for p in params_info:
            self.npars += 1
            hyp = HyParamInfo(p.name, p.minval, 
                            p.maxval, p.transform_type,
                            p.scaling_factor)
            self.params.append(hyp)

    def __print__(self):
        print('Parameter vector:\n')
        for pn in self.getParamName():
            print('\t%s\n',pn)

    def getParamName(self):
        return [p.name for p in self.params]

    def true2trans(self, x):
        y = [self.params[i].true2trans(x[i]) 
                for i in range(self.npars)]
        return y

    def trans2true(self, x):
        y = [self.params[i].trans2true(x[i]) 
                for i in range(self.npars)]
        return y

