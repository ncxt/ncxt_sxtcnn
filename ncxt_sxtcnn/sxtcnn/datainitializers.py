import os
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
from .. import volumeblocks
from scipy.signal.windows import triang

from .utils import ensure_dir


class DataProcessor(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def forward(self, data):
        pass

    @abstractmethod
    def backward(self, data):
        pass

    @abstractmethod
    def init_data(self,
            loader,
            working_directory,
            train_idx=None,
            test_idx=None,
            split = None,
            n_samples=None,
            random_seed=None,
            reset=False,
    ):
        '''Abstact method for the data initialization

        Arguments:
            loader {NCXTDBLoader} -- Data loader
            working_directory {etring} -- working directory for the data
            split {float} -- Data Split
        
        Keyword Arguments:
            n_samples {int} -- Maximum number fo samples (default: {None})
            random_seed {None, int, array_like} -- Seed for numpy.random.RandomState (default: {None})
            reset {bool} -- Write data even though folder exists (default: {False})
        '''
        pass


class SingleBlockProcessor(DataProcessor):
    def __init__(self, block_shape=None):
        super().__init__()

        self.block_shape = block_shape
        self.shape = None

    def forward(self, data):

        self.shape = data.shape

        assert self.block_shape is not None, 'Forward needs a defined block_shape'
        if data.ndim == 3:
            shape = self.block_shape
        elif data.ndim == 4:
            shape = [data.shape[0]] + list(self.block_shape)
        else:
            raise ValueError("input data needs to be either 3d or 4d ndarray")

        retval = np.zeros(shape)
        slices = tuple(
            [slice(0, min(b, s)) for b, s in zip(retval.shape, data.shape)])
        retval[slices] = data[slices]

        
        return retval

    def backward(self, data):
        assert self.shape is not None, 'Backward needs a defined shape'
        shape = list(self.shape)
        if data.ndim == 5:
            assert data.shape[0] == 1, 'Batch size must be one'
            data = data[0]
        if data.ndim == 4:
            shape[0] = data.shape[0]

        retval = np.zeros(shape)
        slices = tuple(
            [slice(0, min(b, s)) for b, s in zip(retval.shape, data.shape)])
        retval[slices] = data[slices]
        return retval

    def init_data(
            self,
            loader,
            working_directory,
            train_idx=None,
            test_idx=None,
            split=None,
            n_samples=None,
            random_seed=None,
            reset=False,
    ):
        working_directory = Path(working_directory)
        train = working_directory / "train"
        validation = working_directory / "validation"

        for directory in [train, validation]:
            if os.path.isdir(directory) and not reset:
                print('Data folder already exists')
                return
            ensure_dir(directory)

        if split is not None:
            assert not train_idx and not test_idx, 'Do not use both split and manual index'

            n_samples = len(loader) if n_samples is None else n_samples
            n_split = int(split * n_samples)
            random_idx = np.random.RandomState(random_seed).permutation(
                n_samples)
            train_idx = random_idx[:n_split]
            test_idx = random_idx[n_split:]

        fileindex_train, fileindex_test = 0, 0
        for ind in train_idx:
            sample = loader[ind]
            x = self.forward(sample["input"])
            y = self.forward(sample["target"])

            data_dict = {"x": x, "y": y}
            np.save(train / f"data{fileindex_train}", data_dict)
            fileindex_train += 1

        for ind in test_idx:
            sample = loader[ind]
            x = self.forward(sample["input"])
            y = self.forward(sample["target"])

            data_dict = {"x": x, "y": y}
            np.save(validation / f"data{fileindex_test}", data_dict)
            fileindex_test += 1


class RandomSingleBlockProcessor(DataProcessor):
    def __init__(self, block_shape=None, n_blocks=4):
        super().__init__()

        self.block_shape = block_shape
        self.n_blocks = n_blocks
        self.shape = None

    def forward(self, data, seed=None):
        self.shape = data.shape

        assert self.block_shape is not None, 'Forward needs a defined block_shape'
        if data.ndim == 3:
            shape = self.block_shape
        elif data.ndim == 4:
            shape = [data.shape[0]] + list(self.block_shape)
        else:
            raise ValueError("input data needs to be either 3d or 4d ndarray")

        retval = np.zeros(shape)
        slices = tuple(
            [slice(0, min(b, s)) for b, s in zip(retval.shape, data.shape)])
        retval[slices] = data[slices]

        if seed:
            roll = [b - s for b, s in zip(retval.shape, data.shape)]
            state = np.random.RandomState(seed)
            shifts = [state.randint(r) if r > 0 else 0 for r in roll]
            retval = np.roll(retval, shifts, axis=np.arange(retval.ndim))

        return retval

    def backward(self, data):
        assert self.shape is not None, 'Backward needs a defined shape'
        shape = list(self.shape)
        if data.ndim == 5:
            assert data.shape[0] == 1, 'Batch size must be one'
            data = data[0]
        if data.ndim == 4:
            shape[0] = data.shape[0]

        retval = np.zeros(shape)
        slices = tuple(
            [slice(0, min(b, s)) for b, s in zip(retval.shape, data.shape)])
        retval[slices] = data[slices]
        return retval

    def init_data(
            self,
            loader,
            working_directory,
            train_idx=None,
            test_idx=None,
            split=None,
            n_samples=None,
            random_seed=None,
            reset=False,
    ):
        working_directory = Path(working_directory)
        train = working_directory / "train"
        validation = working_directory / "validation"

        for directory in [train, validation]:
            if os.path.isdir(directory) and not reset:
                print('Data folder already exists')
                return
            ensure_dir(directory)

        if split is not None:
            assert not train_idx and not test_idx, 'Do not use both split and manual index'

            n_samples = len(loader) if n_samples is None else n_samples
            n_split = int(split * n_samples)
            random_idx = np.random.RandomState(random_seed).permutation(
                n_samples)
            train_idx = random_idx[:n_split]
            test_idx = random_idx[n_split:]

        fileindex_train, fileindex_test = 0, 0
        for ind in train_idx:
            print(f'train ind {ind}')
            sample = loader[ind]
            for seed in range(self.n_blocks):
                x = self.forward(sample["input"], seed)
                y = self.forward(sample["target"], seed)

                data_dict = {"x": x, "y": y}
                np.save(train / f"data{fileindex_train}", data_dict)
                fileindex_train += 1

        for ind in test_idx:
            print(f'test ind {ind}')
            sample = loader[ind]

            for seed in range(self.n_blocks):
                x = self.forward(sample["input"], seed)
                y = self.forward(sample["target"], seed)

                data_dict = {"x": x, "y": y}
                np.save(validation / f"data{fileindex_test}", data_dict)
                fileindex_test += 1


class VolumeBlockProcessor(DataProcessor):
    def __init__(self, block_shape, binning, sampling=1.0):
        super().__init__()
        self.sampling = sampling
        self.block_shape = block_shape
        self.binning = binning

        self.shape = None

    def forward_target(self, data):
        pass
        # TODO: fix forward branch for target data

    def forward(self, data):
        self.shape = data.shape
        block_shape_big = [s * self.binning for s in self.block_shape]
        lac_blocks = volumeblocks.split(
            data,
            block_shape_big,
            binning=self.binning,
            sampling=self.sampling)
        return lac_blocks

    def backward(self, data):
        assert self.shape is not None, 'Backward needs a defined shape'
        ret_dim = data.shape[1]
        retval_shape = [ret_dim] + list(self.shape)[-3:]
        return volumeblocks.fuse(
            data,
            retval_shape,
            binning=self.binning,
            sampling=self.sampling,
            windowfunc=triang)

    def init_data(
            self,
            loader,
            working_directory,
            train_idx=None,
            test_idx=None,
            split=None,
            n_samples=None,
            random_seed=None,
            reset=False,
    ):
        working_directory = Path(working_directory)
        train = working_directory / "train"
        validation = working_directory / "validation"

        for directory in [train, validation]:
            if os.path.isdir(directory) and not reset:
                print(f'Data folder {directory} already exists')
                return
            ensure_dir(directory)

        if split is not None:
            assert not train_idx and not test_idx, 'Do not use both split and manual index'

            n_samples = len(loader) if n_samples is None else n_samples
            n_split = int(split * n_samples)
            random_idx = np.random.RandomState(random_seed).permutation(
                n_samples)
            train_idx = random_idx[:n_split]
            test_idx = random_idx[n_split:]

        block_shape_big = [s * self.binning for s in self.block_shape]
        fileindex_train, fileindex_test = 0, 0

        for ind in train_idx:
            sample = loader[ind]
            data = sample["input"]
            labels = sample["target"]

            blocks_x = volumeblocks.split(
                data,
                block_shape_big,
                binning=self.binning,
                sampling=self.sampling)

            blocks_y = volumeblocks.split_label(
                labels,
                block_shape_big,
                binning=self.binning,
                sampling=self.sampling)

            for bx, by in zip(blocks_x, blocks_y):
                data_dict = {"x": bx, "y": by}
                np.save(train / f"data{fileindex_train}", data_dict)
                fileindex_train += 1

        for ind in test_idx:
            sample = loader[ind]
            data = sample["input"]
            labels = sample["target"]

            blocks_x = volumeblocks.split(
                data,
                block_shape_big,
                binning=self.binning,
                sampling=self.sampling)

            blocks_y = volumeblocks.split_label(
                labels,
                block_shape_big,
                binning=self.binning,
                sampling=self.sampling)

            for bx, by in zip(blocks_x, blocks_y):
                data_dict = {"x": bx, "y": by}
                np.save(validation / f"data{fileindex_test}", data_dict)
                fileindex_test += 1


class RandomBlockProcessor(DataProcessor):
    def __init__(self, block_shape, binning, sampling=1.0):
        super().__init__()
        self.block_shape = block_shape
        self.binning = binning
        self.sampling = sampling

        self.shape = None

    def forward(self, data):
        self.shape = data.shape
        block_shape_big = [s * self.binning for s in self.block_shape]
        lac_blocks = volumeblocks.split(
            data,
            block_shape_big,
            binning=self.binning,
            sampling=self.sampling)
        return lac_blocks

    def backward(self, data):
        assert self.shape is not None, 'Backward needs a defined shape'
        ret_dim = data.shape[1]
        retval_shape = [ret_dim] + list(self.shape)[-3:]
        return volumeblocks.fuse(
            data,
            retval_shape,
            binning=self.binning,
            sampling=self.sampling,
            windowfunc=triang)

    def init_data(
            self,
            loader,
            working_directory,
            train_idx=None,
            test_idx=None,
            split=None,
            n_samples=None,
            n_blocks=1,
            random_seed=None,
            reset=False,
    ):
        working_directory = Path(working_directory)
        train = working_directory / "train"
        validation = working_directory / "validation"

        for directory in [train, validation]:
            if os.path.isdir(directory) and not reset:
                print(f'Data folder {directory} already exists')
                return
            ensure_dir(directory)

        if split is not None:
            assert not train_idx and not test_idx, 'Do not use both split and manual index'

            n_samples = len(loader) if n_samples is None else n_samples
            n_split = int(split * n_samples)
            random_idx = np.random.RandomState(random_seed).permutation(
                n_samples)
            train_idx = random_idx[:n_split]
            test_idx = random_idx[n_split:]

        block_shape_big = [s * self.binning for s in self.block_shape]
        fileindex_train, fileindex_test = 0, 0

        for ind in train_idx:
            seed = ind
            sample = loader[ind]
            data = sample["input"]
            labels = sample["target"]

            blocks_x = volumeblocks.random_blocks(
                data,
                block_shape_big,
                max_patches=n_blocks,
                binning=self.binning,
                random_state=seed)

            blocks_y = volumeblocks.random_blocks_label(
                labels,
                block_shape_big,
                max_patches=n_blocks,
                binning=self.binning,
                random_state=seed)

            for bx, by in zip(blocks_x, blocks_y):
                data_dict = {"x": bx, "y": by}
                np.save(train / f"data{fileindex_train}", data_dict)
                fileindex_train += 1

        for ind in test_idx:
            seed = ind + 1
            sample = loader[ind]
            data = sample["input"]
            labels = sample["target"]

            # blocks_x = volumeblocks.split(
            #     data,
            #     block_shape_big,
            #     binning=self.binning,
            #     sampling=self.sampling)

            # blocks_y = volumeblocks.split_label(
            #     labels,
            #     block_shape_big,
            #     binning=self.binning,
            #     sampling=self.sampling)
            
            blocks_x = volumeblocks.random_blocks(
                data,
                block_shape_big,
                max_patches=self.n_blocks,
                binning=self.binning,
                random_state=seed)

            blocks_y = volumeblocks.random_blocks_label(
                labels,
                block_shape_big,
                max_patches=self.n_blocks,
                binning=self.binning,
                random_state=seed)

            for bx, by in zip(blocks_x, blocks_y):
                data_dict = {"x": bx, "y": by}
                np.save(validation / f"data{fileindex_test}", data_dict)
                fileindex_test += 1

class RandomBlockProcessorRefineMask(DataProcessor):
    def __init__(self, network, block_shape, binning, n_blocks=10, sampling=1.0):
        super().__init__()
        self.network = network
        self.block_shape = block_shape
        self.binning = binning
        self.n_blocks = n_blocks
        self.sampling = sampling
        self.shape = None

    def forward(self, data):
        self.shape = data.shape
        block_shape_big = [s * self.binning for s in self.block_shape]
        lac_blocks = volumeblocks.split(
            data,
            block_shape_big,
            binning=self.binning,
            sampling=self.sampling)
        return lac_blocks

    def backward(self, data):
        assert self.shape is not None, 'Backward needs a defined shape'
        ret_dim = data.shape[1]
        retval_shape = [ret_dim] + list(self.shape)[-3:]
        return volumeblocks.fuse(
            data,
            retval_shape,
            binning=self.binning,
            sampling=self.sampling,
            windowfunc=triang)

    def init_data(
            self,
            loader,
            working_directory,
            train_idx=None,
            test_idx=None,
            split=None,
            n_samples=None,
            random_seed=None,
            reset=False,
    ):

        print('RandomBlockProcessorRefineMask init')
        working_directory = Path(working_directory)
        train = working_directory / "train"
        validation = working_directory / "validation"

        for directory in [train, validation]:
            if os.path.isdir(directory) and not reset:
                print(f'Data folder {directory} already exists')
                return
            ensure_dir(directory)

        if split is not None:
            assert not train_idx and not test_idx, 'Do not use both split and manual index'

            n_samples = len(loader) if n_samples is None else n_samples
            n_split = int(split * n_samples)
            random_idx = np.random.RandomState(random_seed).permutation(
                n_samples)
            train_idx = random_idx[:n_split]
            test_idx = random_idx[n_split:]

        block_shape_big = [s * self.binning for s in self.block_shape]
        fileindex_train, fileindex_test = 0, 0

        for ind in train_idx:
            print(f'train {ind}')
            
            sample = loader[ind]
            data = sample["input"]
            labels = sample["target"]
            cell_mask = self.network.apply(data, probability=True)
            data_in = np.stack((data[0],cell_mask[1]),axis = 0)
            
            blocks_x = volumeblocks.random_blocks(
                data_in,
                block_shape_big,
                max_patches=self.n_blocks,
                binning=self.binning,
                random_state=ind)
            
            blocks_y = volumeblocks.random_blocks_label(
                labels,
                block_shape_big,
                max_patches=self.n_blocks,
                binning=self.binning,
                random_state=ind)

            for bx, by in zip(blocks_x, blocks_y):
                data_dict = {"x": bx, "y": by}
                np.save(train / f"data{fileindex_train}", data_dict)
                fileindex_train += 1

        for ind in test_idx:
            print(f'test_idx {ind}')
            sample = loader[ind]
            data = sample["input"]
            labels = sample["target"]
            cell_mask = self.network.apply(data, probability=True)
            data_in = np.stack((data[0],cell_mask[1]),axis = 0)

            blocks_x = volumeblocks.split(
                data_in,
                block_shape_big,
                binning=self.binning,
                sampling=self.sampling)

            blocks_y = volumeblocks.split_label(
                labels,
                block_shape_big,
                binning=self.binning,
                sampling=self.sampling)

            for bx, by in zip(blocks_x, blocks_y):
                data_dict = {"x": bx, "y": by}
                np.save(validation / f"data{fileindex_test}", data_dict)
                fileindex_test += 1



#  TODO: fix
class FeatureBlockProcessor(DataProcessor):
    def __init__(
            self,
            network,
            block_shape,
            binning,
            sampling=1.0,
    ):
        super().__init__()
        self.network = network
        self.block_shape = block_shape
        self.binning = binning
        self.sampling = sampling

        self.shape = None

    def forward_target(self, data):
        self.shape = data.shape
        block_shape_big = [s * self.binning for s in self.block_shape]

        return volumeblocks.split_label(
            data,
            block_shape_big,
            binning=self.binning,
            sampling=self.sampling)

    def forward(self, data):
        if data.ndim == 3:
            # target data
            return self.forward_target(data)

        features = self.network.features(data)
        input_data = np.concatenate((data, features), axis=0)
        self.shape = data.shape

        block_shape_big = [s * self.binning for s in self.block_shape]
        lac_blocks = volumeblocks.split(
            input_data,
            block_shape_big,
            binning=self.binning,
            sampling=self.sampling)
        return lac_blocks

    def backward(self, data):
        assert self.shape is not None, 'Backward needs a defined shape'
        ret_dim = data.shape[1]
        retval_shape = [ret_dim] + list(self.shape)[-3:]
        return volumeblocks.fuse(
            data,
            retval_shape,
            binning=self.binning,
            sampling=self.sampling,
            windowfunc=triang)

    def init_data(
            self,
            loader,
            working_directory,
            split,
            n_samples=None,
            random_seed=None,
            reset=False,
    ):
        working_directory = Path(working_directory)
        train = working_directory / "train"
        validation = working_directory / "validation"

        for directory in [train, validation]:
            if os.path.isdir(directory) and not reset:
                print(f'Data folder {directory} already exists')
                return
            ensure_dir(directory)

        n_samples = len(loader) if n_samples is None else n_samples
        n_split = int(split * n_samples)
        print(f'init_data loader_len {n_samples}')
        random_idx = np.random.RandomState(random_seed).permutation(n_samples)

        fileindex_train, fileindex_test = 0, 0

        for ind in random_idx[:n_split]:
            sample = loader[ind]
            blocks_x = self.forward(sample["input"])
            blocks_y = self.forward(sample["target"])

            for bx, by in zip(blocks_x, blocks_y):
                data_dict = {"x": bx, "y": by}
                np.save(train / f"data{fileindex_train}", data_dict)
                fileindex_train += 1

        for ind in random_idx[n_split:]:
            sample = loader[ind]
            blocks_x = self.forward(sample["input"])
            blocks_y = self.forward(sample["target"])

            for bx, by in zip(blocks_x, blocks_y):
                data_dict = {"x": bx, "y": by}
                np.save(validation / f"data{fileindex_test}", data_dict)
                fileindex_test += 1
