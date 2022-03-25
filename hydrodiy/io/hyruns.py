import re, json
import copy
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



class SiteBatch():
    """ Class to handle site batches """
    def __init__(self, siteids, nbatch):
        siteids = np.array(siteids)
        nsites = len(siteids)

        errmsg = "Non unique siteids"
        assert len(np.unique(siteids)) == nsites, errmsg

        self.siteids = siteids
        self.nsites = nsites
        self.nbatch = nbatch


    def __getitem__(self, ibatch):
        isites = get_batch(self.nsites, self.nbatch, ibatch)
        return self.siteids[isites].tolist()


    def search(self, siteid):
        for ibatch in range(self.nbatch):
            s = self[ibatch]
            if siteid in s:
                return ibatch



class OptionTask():
    def __init__(self, taskid, context, items):
        self.taskid = taskid
        self.context = context
        self.items = items

    def __str__(self):
        txt = f"Task {self.taskid}:\n\tOptions values\n"
        for key, value in self.items.items():
            txt += f"\t\t{key}: {value}\n"
        txt += "\tContext values\n"
        for key, value in self.context.items():
            txt += f"\t\t{key}: {value}\n"
        return txt

    def __getattr__(self, key):
        if key in self.items:
            return self.items[key]
        elif key in self.context:
            return self.context[key]
        elif key == "names":
            return list(self.items.keys())
        else:
            super(self).__getattr__(key)


    def __getitem__(self, key):
        txt = "/".join(self.items.keys())
        txt += "/" + "/".join(self.context.keys())
        errmsg = f"Expected key {key} in {txt}."
        assert key in self.items or key in self.context, errmsg
        if key in self.items:
            return self.items[key]
        return self.context[key]


    def to_dict(self):
        dd = {\
            "taskid": self.taskid, \
            "context": self.context, \
            "items": self.items
        }
        return dd


    @classmethod
    def from_dict(cls, dd):
        return OptionTask(dd["taskid"], \
                    dd["context"], \
                    dd["items"])


    def log(self, logger):
        """ Log task """
        logger.info("")
        logger.info(f"****** TASK {self.taskid} *******")
        for key, value in self.context.items():
            logger.info(f"Context {key}: {value}")

        logger.info("")
        for key, value in self.items.items():
            logger.info(f"Item {key}: {value}")

        logger.info("***********************")
        logger.info("")



class OptionManager():
    """ Object to manage task contexturation built from lists of parameters """

    def __init__(self, name="Task Manager", **kwargs):
        self.name = name
        self.options = {}
        self.tasks = []
        self.context = kwargs


    def __str__(self):
        txt = f"\n*****\n"
        txt += f"Options manager with {len(self.options)} "
        txt += f"options, {len(self.context)} context and {self.ntasks} tasks:\n"
        txt += "\tOptions values\n"
        for key, value in self.options.items():
            txt += f"\t\t{key} ({len(value)} values): {value}\n"

        if len(self.context)>0:
            txt += "\tConfig values\n"
            for key, value in self.context.items():
                txt += f"\t\t{key}: {value}\n"

        return txt


    def __eq__(self, other):
        # Check other is an option manager
        if not isinstance(other, OptionManager):
            return False

        # Check context
        context = self.context
        context_other = other.context
        for k, v in context.items():
            if k in context_other:
                if not context_other[k] == context[k]:
                    return False
            else:
                return False

        # Check options
        options = self.options
        options_other = other.options
        for k, v in options.items():
            if k in options_other:
                if not options_other[k] == options[k]:
                    return False
            else:
                return False

        # Check tasks
        if not self.ntasks == other.ntasks:
            return False

        for task, otask in zip(self.tasks, other.tasks):
            if not task == otask:
                return False

        return True


    @classmethod
    def from_dict(cls, dd):
        opm = OptionManager(dd.get("name", "Task Manager"))
        opm.context = dd.get("context", {})
        opm.options = dd.get("options", {})
        tasks = dd.get("tasks", [])
        for t in tasks:
            to = OptionTask.from_dict(t)
            opm.tasks.append(to.items)

        return opm


    @classmethod
    def from_file(cls, path):
        with open(path, "r") as fo:
            js = json.load(fo)

        return cls.from_dict(js)


    def to_dict(self):
        dd = {"name": self.name, \
                "context": self.context, \
                "options": self.options, \
                "tasks": [self.get_task(taskid).to_dict() \
                                for taskid in range(self.ntasks)]
        }
        return dd


    def save(self, filename):
        """ Save option manager to disk. Overwrite existing one if different."""
        dd = self.to_dict()
        filename = Path(filename)

        if filename.exists():
            opm = OptionManager.from_file(filename)
            if opm != self:
                with filename.open("w") as fo:
                    json.dump(dd, fo, indent=4)
        else:
            with filename.open("w") as fo:
                json.dump(dd, fo, indent=4)


    def from_cartesian_product(self, **kwargs):
        """ Build an option manager from a cartesian product of options """

        self.options = {}
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

            sk = str(k)
            self.options[sk] = v2

        self.tasks = []
        keys = list(self.options.keys())
        for t in prod(*[self.options[sk] for sk in keys]):
            dd = {k: tt for k, tt in zip(keys, t)}
            self.tasks.append(dd)


    @property
    def ntasks(self):
        """ Number of options """
        return len(self.tasks)


    def get_task(self, taskid):
        ntsks = self.ntasks
        errmsg = f"Expected taskid in [0, {ntsks}[, got {taskid}."
        assert taskid>=0 and taskid<ntsks, errmsg

        # Build task object on the fly
        return OptionTask(taskid, self.context, self.tasks[taskid])


    def search(self, **kwargs):
        """ Search options with particular characteristics,
            e.g. month="(1|10)" will search month equal to 1 or 10.
        """
        taskids = []
        txt = "/".join(self.options.keys())
        for taskid, task in enumerate(self.tasks):
            match = []
            for key, val in kwargs.items():
                skey = str(key)
                errmsg = f"Expected option '{skey}' in {txt}"
                assert skey in self.options, errmsg

                if re.search(val, str(task[skey])):
                    match.append(True)
                else:
                    match.append(False)

            if all(match):
                taskids.append(taskid)

        return taskids



