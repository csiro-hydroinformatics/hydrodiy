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
class HyModel:
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
