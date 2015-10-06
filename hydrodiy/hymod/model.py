import math
import numpy as np


# Fix inputs and outputs - five max.
# RR:
#   inputs = P, PE, T, ??, ??
# RO:
#   inputs = Qup, Div, DS, ??, ??
#
# nodes/link network : 
# [node_id, flow_to, reach_length, x, y, RR_model, RO_model]
class Model:

    def __init__(self, name, nuh, nparams):
        self.name = name
        self.nparams = nparams
        self.uh = np.zeros(nuh).astype(np.float64)

    def __str__(self):
        str = '\n%s model implementation\n' % self.name.upper()

    def setoutputs(self, nval, nout=1):

        self.nout = nout
        self.outputs = np.zeros((nval, nout)).astype(np.float64)
        self.uh = np.zeros(nuh).astype(np.float64)
        self.statesini = np.zeros(nstates).astype(np.float64)
        self.statesuhini = np.zeros(nuh).astype(np.float64)


    def setparams(self, params):
        # Set params value
        self.params = np.array(params).astype(np.float64)

        # Set uh
        nuh_optimised = np.zeros(2).astype(np.int32)
        c_hymod.gr4j_getuh(self.params[3], nuh_optimised, self.uh)
        self.nuh = nuh_optimised[0]


    def setstates(self, statesini=None, statesuhini=None):
        if statesini is None:
            statesini = [self.params[0]/2, self.params[2]/2]

        if statesuhini is None:
            statesuhini = [0.] * self.nuh

        ns = len(statesini)
        self.statesini[:ns] = np.array(statesini).astype(np.float64)

        statesuhini = np.array(statesuhini[:self.nuh]).astype(np.float64)
        self.statesuhini[:self.nuh] = statesuhini


    def run(self, inputs):
        ierr = c_hymod.gr4j_run(self.nuh, self.params, self.uh, 
            inputs, 
            self.statesuhini, 
            self.statesini,
            self.outputs)

        if ierr == esize:
            raise GR4JKernelSizeException(('gr4j_run returns a '
                'size exception %d') % ierr)
        if ierr > 0:
            raise GR4JKernelException(('gr4j_run returns an '
                'exception %d') % ierr)

    
    def getoutputs(self):
        outputs = pd.DataFrame(self.outputs)

        cols = ['Q[mm/d]', 'ECH[mm/d]', 
           'E[mm/d]', 'PR[mm/d]', 
           'QR[mm/d]', 'QD[mm/d]',
           'PERC[mm/d]', 'S[mm]', 'R[mm]']

        outputs.columns = cols[:self.nout]

        return outputs


    def calib(self, inputs, outputs):
        pass



    def __init__(self, name,
            nodes):
        # Model configuation
        self.nodes = nodes

        # Model parameters
        # ... change this. Put more description into param
        # ... yes but a model should bring its own parameters
        self.params = HyParamVector(params_info)

        # inputs and states
        self.inputs_name = inputs_name
        self.states_name = states_name
        self.states_value = np.zeros(len(self.states_name))

    def initialise(self, states):
        self.states = states

    def runloop(self, params_value, inputs, 
            idx_start, idx_end):
        # run model from C here
        a = 0

    def plot(self, ax):
        ax.plot(nodes['x'], nodes['y']) 
