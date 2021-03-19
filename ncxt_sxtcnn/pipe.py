import ncxtamira
from .database import AmiraDatabase
from .sxtcnn.loaders import AmiraLoaderOrganelle
from .sxtcnn.models import UNet3D
from .sxtcnn.processors import RandomBlockProcessor
from .sxtcnn.criteria import CrossEntropyLoss_DiceLoss, CrossEntropyLoss
from .sxtcnn import SXTCNN
from .sxtcnn.utils import hashvars, stablehash, getbestgpu, kfold
from pathlib import Path
import seaborn as sns
import pandas as pd
from tqdm.auto import tqdm
import numpy as np

from ncxtamira.organelles import Organelles


class NCXTPipe:
    def __init__(
        self,
        folder,
        working_directory,
        folder_base=None,
        loader=AmiraLoaderOrganelle,
        model=UNet3D,
        processor=RandomBlockProcessor,
        criterion=CrossEntropyLoss,
        labels=None,
        organelles=None,
        loader_args=None,
        processor_args=None,
        model_args=None,
        criterion_args=None,
        settings=None,
        fold=3,
        sanitize=True,
    ):

        self.working_directory = working_directory
        self.db = AmiraDatabase(folder=folder, wd=working_directory, sanitize=sanitize)
        self.db_base = AmiraDatabase(
            folder=folder_base, wd=working_directory, sanitize=sanitize
        )

        assert bool(labels) != bool(organelles), "Define either organelles or labels"

        if organelles:
            self.task = Organelles.extract_materials(organelles)
        elif labels:
            self.task = labels

        if loader_args is not None:
            self._loader_args = loader_args
        else:
            self._loader_args = dict()

        if processor_args is not None:
            self._processor_args = processor_args
        else:
            self._processor_args = dict()

        if model_args is not None:
            self._model_args = model_args
        else:
            self._model_args = dict()

        if criterion_args is not None:
            self._criterion_args = criterion_args
        else:
            self._criterion_args = dict()

        if settings is not None:
            self.settings = settings
        else:
            self.settings = dict()

        # Fix obligatory arguments

        filelist = self.db.filelist(*self.task) + self.db_base.filelist(*self.task)
        print(
            f"Files {len(self.db.filelist(*self.task))} + {len(self.db_base.filelist(*self.task))}"
        )

        if not self._loader_args.get("files"):
            self._loader_args["files"] = filelist
        if not self._loader_args.get("features"):
            self._loader_args["features"] = self.task
        if not self._loader_args.get("sanitize"):
            self._loader_args["sanitize"] = sanitize
        if not self._model_args.get("num_classes"):
            self._model_args["num_classes"] = len(self.task) + 1
        self._criterion_args["ignore_index"] = self._model_args["num_classes"]

        self.fold = fold

        self.loader = loader(**self._loader_args)
        self.processor = processor(**self._processor_args)
        self.model = model(**self._model_args)
        self.criterion = criterion(**self._criterion_args)

        self.sxtcnn = None
        self._device = None

    def dataframe(self):
        return self.db.dataframe_sel(*self.task)

    def dataframe_base(self):
        return self.db_base.dataframe_sel(*self.task)

    @property
    def device(self):
        return self._device

    @device.setter
    def device(self, value):
        if value == "cuda":
            self._device = f"cuda:{getbestgpu()}"
        else:
            self._device = value

    def kfold_split(self, index):
        len_task = len(self.db.filelist(*self.task))
        len_base = len(self.db_base.filelist(*self.task))
        train_idx, valid_idx = kfold(index, self.fold, len_task)
        train_idx = list(train_idx) + [i + len_task for i in range(len_base)]
        return train_idx, valid_idx

    def setup(self):
        self.sxtcnn = SXTCNN(
            self.loader,
            self.processor,
            self.model,
            self.criterion,
            self.working_directory,
            self.settings,
        )
        self.sxtcnn.set_device(self._device)

    def init_fold(self, index):
        train_idx, valid_idx = self.kfold_split(index)
        self.sxtcnn.init_data(train_idx, valid_idx)

    def train(self):
        if not self.sxtcnn:
            self.setup()

        # for f in self.loader.files:
        #     print(Path(f).name)

        if self.fold == 0:
            self.init_fold(0)
            self.sxtcnn.load_trained()

        for i in range(self.fold):
            self.init_fold(i)
            self.sxtcnn.load_trained()

    def plot_train(self, index=0):
        self.init_fold(index)
        self.sxtcnn.logger.plot()

    def model_summary(self):
        if not self.sxtcnn:
            self.setup()

        shape = self.sxtcnn.processor.block_shape
        in_channels = self.sxtcnn.model.in_channels

        temp_device = self.sxtcnn.device
        self.sxtcnn.set_device("cpu")
        self.sxtcnn.model.summary((in_channels, *shape))
        self.sxtcnn.set_device(temp_device)
        return

    def check_database(self, sampling=100, dim=0):
        rows = []
        for si, data in enumerate(tqdm(self.loader)):
            input = data["input"]
            target = data["target"]
            key = data["key"]
            vol = input[dim]

            for k, v in key.items():
                data = vol[target == v]
                for d in np.random.choice(data, sampling):
                    row = {"sample": si, "key": k, f"value": d}
                    rows.append(row)

        df = pd.DataFrame(rows)

        g = sns.catplot(
            data=df,
            kind="bar",
            x="key",
            y="value",
            hue="sample",
            #     ci="sd",
            palette="dark",
            alpha=0.6,
            height=6,
            aspect=8 / 5,
        )
        g.despine(left=True)
        g.set_axis_labels("", "Mean LAC")

    def check_loader(self, index=0):
        proj = ncxtamira.AmiraCell.from_hx(
            self.loader.files[index], sanitize=self._loader_args["sanitize"]
        )
        proj.preview()

        data = self.loader[index]
        proj_loader = ncxtamira.AmiraCell(data["input"][0], data["target"], data["key"])
        proj_loader.preview()
