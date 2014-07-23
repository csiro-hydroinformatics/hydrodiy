import math
import numpy as np
import cPickle as pickle

from gr4j_kernel import adaptor, kernel

class GR4J(object):
    """GR4J model for daily rainfall-runoff simulation.

    Parameters:
        X1    Capacity of the production store (mm)
        X2    Water exchange coefficient (mm)
        X3    Capacity of the routing store (mm)
        X4    Time base of the unit hydrograph (days)

    Internal Storages:
        Sp    Level in production store (mm), a scalar
        Sr    Level in routing store (mm), a scalar
        V1    Levels in the first unit hydrograph routing (mm), a vector
        V2    Levels in the second unit hydrograph routing (mm), a vector
    """
    def __init__(self, name=""):
        self.name = name

        # The time base of each unit hydrograph routing.
        # The time bases are updated whenever a new X4 parameter is set.
        self.__nUH1 = 0
        self.__nUH2 = 0

        # Create parameter and state arrays.
        self._V = np.zeros(2+3*kernel.nh, dtype=np.float)
        self._X = np.zeros(kernel.npx+3*kernel.nh, dtype=np.float)

        # Set the same default parameter values of TIME version.
        # Use properties to do all checks and time base setting.
        self.X1 = 350.0
        self.X2 = 0.0
        self.X3 = 40.0
        self.X4 = 0.5

        self.init()

    def init(self, P=None, Q=None):
        """Initialize internal storages and calculate internal variables,
        including unit hydrograph ordinates. You should call this method
        whenever you change parameter values.
        If P or Q is not given, two internal storages are set to zero.
        If both P and Q are given, two internall storages are set to
        proper values for the given P and Q values. If you give long-term
        average values, the model will be set to allow faster initialisation
        """
        if P is None or Q is None:
            adaptor.initzero(self._V)
        else:
            kernel.init(self._V, self._X, P, Q)

        kernel.uh1(self._X)
        kernel.uh2(self._X)

    # Parameters
    @property
    def X1(self):
        """Capacity of the production store (mm)"""
        return self._X[0]

    @X1.setter
    def X1(self, value):
        if not np.isfinite(value):
            raise ValueError("Attempt to set parameter to NaN value")
        elif value < 0.0:
            raise ValueError("Negative value, %.2f."%value)
        elif value > 2000.0:
            raise ValueError("%.2f higher than 1500.0 (mm)"%value)
        self._X[0] = value

    @property
    def X2(self):
        """Water exchange coefficient (mm)"""
        return self._X[1]

    @X2.setter
    def X2(self, value):
        if not np.isfinite(value):
            raise ValueError("Attempt to set parameter to NaN value")
        elif value < -100.0:
            raise ValueError("%.2f lower than -10.0 (mm)"%value)
        elif value > 100.0:
            raise ValueError("%.2f higher than 5.0 (mm)"%value)
        self._X[1] = value

    @property
    def X3(self):
        """Capacity of the routing store (mm)"""
        return self._X[2]

    @X3.setter
    def X3(self, value):
        if not np.isfinite(value):
            raise ValueError("Attempt to set parameter to NaN value")
        elif value < 1.0:
            raise ValueError("%.2f lower than 1.0 (mm)"%value)
        elif value > 500.0:
            raise ValueError("%.2f higher than 500.0 (mm)"%value)
        self._X[2] = value

    @property
    def X4(self):
        """Time base of the unit hydrograph (days)"""
        return self._X[3]

    @X4.setter
    def X4(self, value):
        if not np.isfinite(value):
            raise ValueError("Attempt to set parameter to NaN value")
        elif value < 0.5:
            raise ValueError("%.2f lower than 0.5 (day)"%value)
        elif value > 10.0:
            raise ValueError("%.2f higher than 4.0 (days)"%value)

        self.__nUH1 = math.ceil(value)
        self.__nUH2 = math.ceil(2.0*value)
        self._X[3] = value

    # Internal storages
    @property
    def Sp(self):
        """Level in production store (mm)"""
        return self._V[0]

    @Sp.setter
    def Sp(self, value):
        if not np.isfinite(value):
            raise ValueError("Attempt to set state to NaN value")
        elif value < 0.0:
            raise ValueError("Negative value, %.4E."%value)
        self._V[0] = value

    @property
    def Sr(self):
        """Level in routing store (mm)"""
        return self._V[1]

    @Sr.setter
    def Sr(self, value):
        if not np.isfinite(value):
            raise ValueError("Attempt to set state to NaN value")
        elif value < 0.0:
            raise ValueError("Negative value, %.4E."%value)
        self._V[1] = value

    @property
    def V1(self):
        """Level in the first unit hydrograph (mm).
        The length of the return array depends on X4 parameter."""
        a = self._V[2:(2+self.__nUH1)].copy()
        a.flags.writeable = False
        return a

    @V1.setter
    def V1(self, value):
        try:
            value[0]
        except TypeError:
            raise ValueError("V1 should be a vector, not a scalar.")

        if np.any(~np.isfinite(value)):
            raise ValueError("Attempt to set state to NaN value")
        elif len(value) != self.__nUH1:
            raise ValueError("V1 should have a length, %d."%self.__nUH1)

        self._V[2:(2+self.__nUH1)] = value
        self._V[(2+self.__nUH1):(kernel.nh+2)] = 0.0

    @property
    def V2(self):
        """Level in the first unit hydrograph (mm)
        The length of the return array depends on X4 parameter."""
        a = self._V[(kernel.nh+2):(kernel.nh+2+self.__nUH2)].copy()
        a.flags.writeable = False
        return a

    @V2.setter
    def V2(self, value):
        try:
            value[0]
        except TypeError:
            raise ValueError("V2 should be a vector, not a scalar.")

        if np.any(~np.isfinite(value)):
            raise ValueError("Attempt to set state to NaN value")
        elif len(value) != self.__nUH2:
            raise ValueError("V2 should have a length, %d."%self.__nUH2)

        self._V[(kernel.nh+2):(kernel.nh+2+self.__nUH2)] = value
        self._V[(kernel.nh+2+self.__nUH2):] = 0.0

    def run(self, P, PET):
        """
        P    Precipitation, a scalar or 1-D array (mm)
        PET  Potential evapotranspiration, a scalar or 1-D array (mm)

        Returns streamflow outcome, Sp and Sr.

        Internal storages remember the values at the end of a simulation.

        The length of P should be same as the length of PET.
        If not, an exception is raised.

        Even in the case P or PET is a scalar,
        Q is always a 1-D array with a single value.
        So, type Q[0] to retrieve a scalar value for streamflow outcome.
        """
        P = np.array(P, dtype=np.float, copy=False, ndmin=1)
        PET = np.array(PET, dtype=np.float, copy=False, ndmin=1)

        if len(P.shape) != 1:
            raise ValueError("P should be a scalar or 1-D array.")

        if P.shape != PET.shape:
            raise ValueError("P and PET have different shapes.")

        Q = np.zeros(P.shape, dtype=np.float)

        adaptor.run(self._V, self._X, P, PET, Q)

        return Q

    def dump(self, file):
        """Dump internal states to a file.

        file    file-like open handle

        Example:

        model.dump(open("states.pkl", "w"))
        """
        states = {"X": self._X, "V": self._V}
        pickle.dump(states, file)

    def load(self, file):
        """Read and restore internal states from a file.

        file    file-like open handle

        Example:

        model.load(open("states.pkl"))
        """
        states = pickle.load(file)
        self._X = states["X"]
        self._V = states["V"]

    def __del__(self):
        pass

    def __str__(self):
        return "\n".join([
            "Model: GR4J",
            "Version: 1.01",
            "Name: %s"%self.name,
            "X1: %.4E (mm)"%self.X1,
            "X2: %.4E (mm)"%self.X2,
            "X3: %.4E (mm)"%self.X3,
            "X4: %.4E (days)"%self.X4,
            "Sp: %.4E (mm)"%self.Sp,
            "Sr: %.4E (mm)"%self.Sr,
            "V1: %s (mm)"%self.V1,
            "V2: %s (mm)"%self.V2,
            ])
