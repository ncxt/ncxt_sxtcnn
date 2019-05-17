import numpy as np

from .models import UNet3D, RefUNet3D
from .dataloaders import NCXTDBLoader

from .datainitializers import SingleBlockProcessor, VolumeBlockProcessor
from .datainitializers import RandomBlockProcessor, FeatureBlockProcessor

from .sxt_cnn_wrapper import SXT_CNN_WRAPPER

import hashlib


def stablehash(*arglist):
    identifier = (".").join([str(x) for x in arglist])
    return int(hashlib.sha256(identifier.encode("utf-8")).hexdigest(), 16) % 2 ** 16


def kfold(index, k, n_samples, random_seed=1):
    assert index < k, "Index must be less then k"
    random_idx = np.random.RandomState(random_seed).permutation(n_samples)

    folds = np.array_split(random_idx, k)

    valid = folds[index]
    folds.pop(index)

    train = [item for sublist in folds for item in sublist]

    return train, valid


import torch.nn as nn
from .datainitializers import DataProcessor


class SegFactory:
    def __init__(self, database=None, working_directory=None, **kwargs):
        self.database = database
        self.working_directory = working_directory

        self.loader = None
        self.model = None
        self.processor = None

        cellmask = kwargs["cellmask"]
        features = kwargs["features"]

        # defualt arguments
        self.ignore_index = 1 + cellmask + len(features)
        self.loader_args = {"cellmask": cellmask, "features": features}
        self.model_args = {
            "num_classes": 1 + cellmask + len(features),
            "in_channels": 1,
        }
        self.processor_args = dict()
        self.seg_args = dict()

    def set_loader(self, cls, **kwargs):
        self.loader_args.update(kwargs)
        self.loader = cls

    def set_model(self, cls, **kwargs):
        assert issubclass(cls, nn.Module), "Class should be a torch Module"
        self.model_args.update(kwargs)
        self.model = cls

    def set_processor(self, cls, **kwargs):
        assert issubclass(cls, DataProcessor), "Class should be a torch DataProcessor"
        self.processor_args.update(kwargs)
        self.processor = cls

    def set_segargs(self, **kwargs):
        self.seg_args.update(kwargs)

    def __call__(self, **kwargs):
        assert self.loader is not None, "Undefined loader"
        assert self.model is not None, "undefined model"
        assert self.processor is not None, "undefined processor"
        arg_init = kwargs.get("init", False)

        loader = self.loader(self.database, **self.loader_args)
        model = self.model(**self.model_args)
        processor = self.processor(**self.processor_args)

        dataidentifier = stablehash(self.loader_args, self.processor_args)
        cnnidentifier = stablehash(self.model_args)

        name = f"SXT_CNN_{type(model).__name__}_{cnnidentifier}"

        seg_params = {"name": name, "ignore": self.ignore_index}
        self.seg_args.update(seg_params)
        if self.working_directory:
            wd = f"{self.working_directory}/data_{type(processor).__name__}_{dataidentifier}/"
            self.seg_args["working_directory"] = wd

        seg = SXT_CNN_WRAPPER(loader, model, processor, self.seg_args)
        if arg_init:
            init_kwargs = dict(kwargs)
            init_kwargs.pop("init")
            train, test = (
                [i for i in range(len(loader))],
                [i for i in range(len(loader))],
            )
            seg.init_data(train_idx=train, test_idx=test, **init_kwargs)

        return seg

    def kfold(self, index, k, **kwargs):
        assert self.loader is not None, "Undefined loader"
        assert self.model is not None, "undefined model"
        assert self.processor is not None, "undefined processor"
        arg_init = kwargs.get("init", False)

        loader = self.loader(self.database, **self.loader_args)
        model = self.model(**self.model_args)
        processor = self.processor(**self.processor_args)

        train, test = kfold(index, k, len(loader))
        fold_args = {"train": train, "test": test}

        dataidentifier = stablehash(fold_args, self.loader_args, self.processor_args)
        cnnidentifier = stablehash(self.model_args)

        name = f"SXT_CNN_{type(model).__name__}_{cnnidentifier}"

        seg_params = {"name": name, "ignore": self.ignore_index}
        self.seg_args.update(seg_params)
        if self.working_directory:
            wd = f"{self.working_directory}/data_{type(processor).__name__}_{dataidentifier}/"
            self.seg_args["working_directory"] = wd

        print(f"K-fold ({k}) training on train {train} test {test} ")

        seg = SXT_CNN_WRAPPER(loader, model, processor, self.seg_args)
        seg.train_idx = train
        seg.test_idx = test
        if arg_init:
            print("Init kfold")
            init_kwargs = dict(kwargs)
            init_kwargs.pop("init")
            seg.init_data(**init_kwargs)

        return seg

    def asdict(self):
        retval = dict()
        retval["loader"] = self.loader.__name__
        retval["model"] = self.model.__name__
        retval["processor"] = self.processor.__name__

        retval["task_args"] = {
            "cellmask": self.loader_args["cellmask"],
            "features": self.loader_args["features"],
        }
        export_loader_args = dict(self.loader_args)
        export_loader_args.pop("cellmask")
        export_loader_args.pop("features")

        export_model_args = dict(self.model_args)
        export_model_args.pop("num_classes")
        export_model_args.pop("in_channels")
        retval["loader_args"] = export_loader_args
        retval["model_args"] = export_model_args
        retval["processor_args"] = self.processor_args

        return retval

    def savedict(self, path):
        with open(path, "w") as fp:
            json.dump(self.asdict(), fp, indent=4)


import json
from . import dataloaders
from . import models
from . import datainitializers


def load_factory_from_json(path):
    with open(path, "r") as fp:
        data = json.load(fp)
    return load_factory_from_dict(data)


def load_factory_from_dict(loaddict):

    loader = getattr(dataloaders, loaddict["loader"])
    model = getattr(models, loaddict["model"])
    processor = getattr(datainitializers, loaddict["processor"])

    factory = SegFactory(None, **loaddict["task_args"])
    factory.set_loader(loader, **loaddict["loader_args"])
    factory.set_model(model, **loaddict["model_args"])
    factory.set_processor(processor, **loaddict["processor_args"])

    return factory


import os


def load_factory(arg):
    is_dict = isinstance(arg, dict)
    is_string = isinstance(arg, str)

    if not any([is_dict, is_string]):
        raise ValueError("Argument must be a dictionary or path to json file")

    if is_dict:
        return load_factory_from_dict(arg)

    if is_string:
        return load_factory_from_json(arg)

