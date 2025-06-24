import re
import json
import time
from pathlib import Path
from itertools import product as prod
import numpy as np

# Specify key names to allow possibily to change this
# and solve backward compatibility issues when
# importing / exporting data to json
#
# !! Do not change this attribute directly, always use the set_dict_keyname
# function
_DICT_KEYNAMES_DEFAULT = {
    "context_name": "context",
    "task_options_name": "options",
    "manager_options_name": "options"
    }

_DICT_KEYNAMES = {}


def reset_dict_keyname():
    """ Reset all key/name pairs for import and export of json data """
    for key, val in _DICT_KEYNAMES_DEFAULT.items():
        _DICT_KEYNAMES[key] = val


# Initial setup of keynames
reset_dict_keyname()


def set_dict_keyname(key, name):
    """ Set a new key/name pair for import and export of json data """
    txt = "/".join(list(_DICT_KEYNAMES_DEFAULT.keys()))
    errmsg = f"Expected key in {txt}, got {key}"
    assert key in _DICT_KEYNAMES_DEFAULT, errmsg
    _DICT_KEYNAMES[key] = name


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
        raise ValueError(f"Expected nelements ({nelements})"
                         + f">= nbatch ({nbatch})")

    if ibatch < 0 or ibatch >= nbatch:
        raise ValueError(f"Expected ibatch ({ibatch}) in"
                         + f" [0, {nbatch-1}].")

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
    def __init__(self, taskid, context, options):
        self.taskid = taskid
        self.context = {} if context is None else context
        self.options = options

    def __str__(self):
        txt = f"Task {self.taskid}:\n\tOptions values\n"
        for key, value in self.options.items():
            txt += f"\t\t{key}: {value}\n"
        txt += "\tContext values\n"
        for key, value in self.context.items():
            txt += f"\t\t{key}: {value}\n"
        return txt

    def __getattr__(self, key):
        if key in self.options:
            return self.options[key]
        elif key in self.context:
            return self.context[key]
        elif key == "names":
            return list(self.options.keys())
        else:
            super(OptionTask, self).__getattribute__(key)

    def __getitem__(self, key):
        txt = "/".join(self.options.keys())
        txt += "/" + "/".join(self.context.keys())
        errmsg = f"Expected key {key} in {txt}."
        assert key in self.options or key in self.context, errmsg
        if key in self.options:
            return self.options[key]
        return self.context[key]

    def to_dict(self, prefix="", include_context=True):
        dd = {
            "taskid": self.taskid,
            _DICT_KEYNAMES["task_options_name"]: self.options
            }
        if include_context:
            dd[_DICT_KEYNAMES["context_name"]] = self.context

        return dd

    @classmethod
    def from_dict(cls, dd):
        return OptionTask(dd["taskid"],
                          dd.get(_DICT_KEYNAMES["context_name"], None),
                          dd[_DICT_KEYNAMES["task_options_name"]])

    def log(self, logger):
        """ Log task """
        logger.info("")
        logger.info(f"****** TASK {self.taskid} *******")
        for key, value in self.context.items():
            logger.info(f"Context {key}: {value}")

        logger.info("")
        for key, value in self.options.items():
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
        txt = "\n*****\n"
        txt += f"Options manager with {len(self.options)} "
        txt += f"options, {len(self.context)} context"\
               + f" and {self.ntasks} tasks:\n"
        txt += "\tOptions values\n"
        for key, value in self.options.items():
            txt += f"\t\t{key} ({len(value)} values): {value}\n"

        if len(self.context) > 0:
            txt += "\tConfig values\n"
            for key, value in self.context.items():
                txt += f"\t\t{key}: {value}\n"

        return txt

    def __getattr__(self, key):
        if key in self.context:
            return self.context[key]
        else:
            super(OptionManager, self).__getattribute__(key)

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
        opm.context = dd.get(_DICT_KEYNAMES["context_name"], {})
        opm.options = dd.get(_DICT_KEYNAMES["manager_options_name"], {})
        tasks = dd.get("tasks", [])
        for t in tasks:
            to = OptionTask.from_dict(t)
            opm.tasks.append(to.options)

        return opm

    @classmethod
    def from_file(cls, path, wait_secs=2):
        with open(path, "r") as fo:
            # Attempt to open the file several times in case
            # it's being written from another process
            js = None
            for attempt in range(2):
                try:
                    js = json.load(fo)
                except json.decoder.JSONDecodeError:
                    time.sleep(wait_secs)

            if js is None:
                raise IOError(f"Cannot read file {path}")

        return cls.from_dict(js)

    def to_dict(self, prefix=""):
        dd = {
            "name": self.name,
            _DICT_KEYNAMES["context_name"]: self.context,
            _DICT_KEYNAMES["manager_options_name"]: self.options,
            "tasks": [self.get_task(taskid).to_dict(prefix,
                                                    include_context=False)
                      for taskid in range(self.ntasks)]
        }
        return dd

    def save(self, filename, prefix="", overwrite=False):
        """ Save option manager to disk.
        Overwrite existing one if different.
        """
        dd = self.to_dict(prefix)
        filename = Path(filename)

        if filename.exists() and not overwrite:
            return

        try:
            with filename.open("w") as fo:
                json.dump(dd, fo, indent=4)
        except Exception:
            pass

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
                errmsg = "Expected an iterable, a float,"\
                         + f" an int or a string, got {type(v)}."
                raise TypeError(errmsg)

            # convert numpy to list
            try:
                v2 = v2.tolist()
            except AttributeError:
                pass

            # Store options
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
        assert taskid >= 0 and taskid < ntsks, errmsg

        # Build task object on the fly
        return OptionTask(taskid, self.context, self.tasks[taskid])

    def search(self, **kwargs):
        """ Search tasks with criteria on options,
            e.g. month="(1|10)" will search month equal to 1 or 10.
        """
        taskids = []
        txt = "/".join(self.options.keys())
        for taskid, task in enumerate(self.tasks):
            match = []
            for key, val in kwargs.items():
                errmsg = f"Expected option '{key}' in {txt}"
                assert key in self.options, errmsg
                s1 = re.sub("\\[|\\]", "", str(val))
                s2 = re.sub("\\[|\\]", "", str(task[key]))
                if re.search(s1, s2):
                    match.append(True)
                else:
                    match.append(False)

            if all(match):
                taskids.append(taskid)

        return taskids

    def find(self, **kwargs):
        """ Find options with specific values for options,
            e.g. month=1 will search month equal to 1.
        """
        kw = {k: f"^{v}$" for k, v in kwargs.items()}
        return self.search(**kw)

    def match(self, other_task, exclude=[], **kwargs):
        """ Find all tasks that match another task excluding certain items.
        """
        # Perform a search first
        if kwargs == {}:
            taskids_initial = list(range(self.ntasks))
        else:
            taskids_initial = self.find(**kwargs)

        txt = "/".join(self.options.keys())
        taskids = []
        for taskid in taskids_initial:
            task = self.tasks[taskid]
            match = True
            for key, val in other_task.options.items():
                # Skip items flagged as such
                if key in exclude:
                    continue

                errmsg = f"Expected option '{key}' in {txt}"
                assert key in task, errmsg

                # Check equality of items
                if not task[key] == val:
                    match = False
                    break

            if match:
                taskids.append(taskid)

        return taskids
