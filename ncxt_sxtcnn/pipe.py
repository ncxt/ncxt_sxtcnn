from .hxdatabase import Database
from .sxtcnn.loaders import AmiraLoaderx100
from .sxtcnn.models import UNet3D
from .sxtcnn.processors import RandomBlockProcessor
from .sxtcnn.criteria import CrossEntropyLoss_DiceLoss, CrossEntropyLoss
from .sxtcnn import SXTCNN
from .sxtcnn.utils import hashvars, stablehash, getbestgpu, kfold
from pathlib import Path


class NCXTPipe:
    def __init__(
        self,
        folder,
        folder_base,
        working_directory,
        loader=AmiraLoaderx100,
        model=UNet3D,
        processor=RandomBlockProcessor,
        criterion=CrossEntropyLoss,
        task=["membrane"],
        loader_args=None,
        processor_args=None,
        model_args=None,
        criterion_args=None,
        settings=None,
        fold=3,
    ):

        self.working_directory = working_directory
        self.db = Database(folder=folder, wd=working_directory, sanitize=True)
        self.db_base = Database(folder=folder_base, wd=working_directory, sanitize=True)

        self.task = task

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
        if not self._loader_args.get("files"):
            self._loader_args["files"] = self.db.filelist(
                *task
            ) + self.db_base.filelist(*task)
        if not self._loader_args.get("features"):
            self._loader_args["features"] = self.task
        if not self._model_args.get("num_classes"):
            self._model_args["num_classes"] = len(self.task) + 1

        self.fold = fold

        self.loader = loader(**self._loader_args)
        self.processor = processor(**self._processor_args)
        self.model = model(**self._model_args)
        self.criterion = criterion(**self._criterion_args)

        self.sxtcnn = None
        self._device = None

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
        train_idx = train_idx + [i + len_task for i in range(len_base)]
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

    def train(self):
        if not self.sxtcnn:
            self.setup()

        # for f in self.loader.files:
        #     print(Path(f).name)

        if self.fold == 0:
            self.sxtcnn.init_kfold(0, 0)
            self.sxtcnn.run()

        for i in range(self.fold):
            train_idx, valid_idx = self.kfold_split(i)
            self.sxtcnn.init_data(train_idx, valid_idx)
            self.sxtcnn.run()

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