
import sys, types

from hymod.model import Model
from hymod.model import ModelError

from hymod.models.gr4j import GR4J
from hymod.models.gr2m import GR2M
from hymod.models.turcmezentsev import TurcMezentsev
from hymod.models.lagroute import LagRoute

def _populate_list():
    
    models = {}
    module = sys.modules[__name__]

    for id, cls in vars(module).iteritems():
        try:
            if issubclass(cls, Model) & (id!='Model'):
                instance = cls()
                models[instance.name] = instance
        except TypeError:
            pass

    return models

model_list = _populate_list()


def get(name):

    if name in model_list:
        return model_list[name]
    else:
        moderr = ModelError(name, 
            ierr = -1, 
            message='Cannot find instance of model {0}'.format(name))
        raise moderr

    
