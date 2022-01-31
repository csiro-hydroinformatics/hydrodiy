import re
from pathlib import Path
from itertools import product as prod
import numpy as np
import pandas as pd

from hydrodiy.io import csv


def get_batch(nelements, nbatch, ibatch):
    """ Returns the indices of elements within a batch

    Parameters
    -----------
    nelements : int
        Number of elements
    nbatch : int
        Number of batches
    ibatch : int
        Batch index (from 0 to nbatch-1)

    Returns
    -----------
    idx : list
        List of integer containing sites indexes (0 = first site)

    Example
    -----------
    >>>  idx = iutils.get_ibatch(20, 5, 1)
    [4, 5, 6, 7]
    >>>  idx = iutils.get_ibatch(20, 5, 2)
    [8, 9, 10, 11]

    """
    if nelements < 1:
        raise ValueError(f"Expected nelements>=1, got {nelements}.")

    if nelements < nbatch:
        raise ValueError(f"Expected nelements ({nelements}) "+\
                            f">= nbatch ({nbatch})")

    if ibatch < 0 or ibatch >= nbatch:
        raise ValueError(f"Expected ibatch ({ibatch}) in "+\
                        f"[0, {nbatch-1}].")

    return np.array_split(np.arange(nelements), nbatch)[ibatch]



class OptionManager():
    """ Object to manage task configuration built from lists of parameters """

    def __init__(self, name="Option Manager", description="List of tasks"):
        self.name = name
        self.description = description
        self.options = {}
        self.tasks = []


    def __str__(self):
        txt = f"\n*****\n"
        txt += f"Options manager with {len(self.options)} "
        txt += f"options and {self.ntasks} tasks:\n"
        txt += "Options values\n"
        for key, value in self.options.items():
            txt += f"\t{key} ({len(value)} values): {value}\n"
        return txt


    @classmethod
    def from_cartesian_product(cls, **kwargs):
        """ Build an option manager from a cartesian product of options """

        opm = OptionManager("Option Manager", \
                        "Tasks built from cartesian product")
        opm.options = {}
        for k, v in kwargs.items():
            if isinstance(v, str) or isinstance(v, float) \
                                    or isinstance(v, int):
                v2 = [v]
            elif hasattr(v, "__iter__"):
                v2 = v
            else:
                errmsg = "Expected an iterable, a float, "+\
                        f"an int or a string, got {type(v)}."
                raise TypeError(errmsg)

            opm.options[k] = v2

        opm.tasks = []
        keys = list(opm.options.keys())
        for t in prod(*[opm.options[k] for k in keys]):
            dd = {k: tt for k, tt in zip(keys, t)}
            opm.tasks.append(dd)

        return opm


    @property
    def ntasks(self):
        """ Number of options """
        return len(self.tasks)


    def get_task(self, taskid):
        ntsks = self.ntasks
        errmsg = f"Expected taskid in [0, {ntsks}[, got {taskid}."
        assert taskid>=0 and taskid<ntsks, errmsg

        # Build task object on the fly
        # with basic properties
        task = type("Task", (),self.tasks[taskid])
        task.names = list(self.options.keys())
        task.get = lambda k: getattr(task, k)

        return task


    def to_dataframe(self):
        df = pd.DataFrame(self.tasks)
        df.index.name = "taskid"
        return df


    def save(self, fout):
        df = self.to_dataframe()
        csv.write_csv(df, fout, \
                f"Content of option manager {self.name}", \
                Path(__file__).resolve(), \
                write_index=True, compress=False)


    def search(self, **kwargs):
        """ Search options with particular characteristics,
            e.g. month="(1|10)" will search month equal to 1 or 10.
        """
        taskids = []
        txt = "/".join(self.options.keys())
        for taskid, task in enumerate(self.tasks):
            match = []
            for key, val in kwargs.items():
                errmsg = f"Expected option '{key}' in {txt}"
                assert key in self.options, errmsg

                if re.search(val, str(task[key])):
                    match.append(True)
                else:
                    match.append(False)

            if all(match):
                taskids.append(taskid)

        return taskids




