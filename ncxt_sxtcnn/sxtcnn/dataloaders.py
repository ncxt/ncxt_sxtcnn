import json
import os

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import curve_fit
from scipy.signal import argrelextrema

from .. import io

from .models import UNet3D
from .utils import bin_ndarray_mode, bin_ndarray_single, crop_to_nonzero


def normalize_nonzero(image):
    p2, p98 = np.percentile(image[image > 0], (2, 98))
    return (image - p2) / float(p98 - p2)


def equalize_nonzero(image, nbr_bins=256):
    im_nonzero = image[image > 0]
    imhist, bins = np.histogram(im_nonzero.flatten(), nbr_bins, density=True)
    cdf = imhist.cumsum()  # cumulative distribution function
    cdf = np.insert(cdf / cdf[-1], 0, 0)

    # use linear interpolation of cdf to find new pixel values
    im2 = image.flatten()
    im2[im2 > 0] = np.interp(im2[im2 > 0], bins, cdf)

    return im2.reshape(image.shape)


def feature_selector(image, keydict, features, cellmask=True):
    # print(f'input {features}')
    keydict = dict((k.lower(), v) for k, v in keydict.items())

    # print(f"input {keydict}")

    cell_labels = [v for k, v in keydict.items() if "ext" not in k]
    ignore_labels = [v for k, v in keydict.items() if "ignore" in k]

    ignore_mask = np.isin(image, ignore_labels).astype(int)
    retval_key = {"exterior": 0}
    label = (np.isin(image, cell_labels).astype(int)
             if cellmask else np.zeros(image.shape, dtype=int))
    if cellmask:
        retval_key["cell"] = 1

    for i, keys in enumerate(features):
        index = cellmask + 1 + i
        # make feature iterable
        if not isinstance(keys, (list, tuple)):
            keys = [keys]
        # print(f'processing {keys}')

        for key in keys:
            # print(f'key {key} set to {index} ')
            try:
                value = keydict[key]
                label[image == value] = index
                retval_key[key] = index
            except KeyError:
                pass

    # print(f"after features {retval_key}")

    if np.sum(ignore_mask):
        index = max(retval_key.values()) + 1
        retval_key["ignore"] = index
        label[ignore_mask > 0] = index

    # print(f"retval {retval_key}")

    return label, retval_key


class NCXTMockLoader:
    def __init__(self, shape, labels, length):
        self.shape = shape
        self.lenght = length
        self.labels = labels

    def __len__(self):
        return self.lenght

    def __getitem__(self, index):
        key = {f'material_{n}': n for n in range(self.labels)}
        random_lac = np.random.random(self.shape)
        random_label = np.random.randint(
            low=0, high=self.labels, size=self.shape, dtype='int')

        return {
            "input": random_lac.reshape(1, *random_lac.shape),
            "target": random_label.astype(int),
            "key": key,
        }

    def __call__(self, data):
        retval = data.copy()
        return retval.reshape(1, *retval.shape)


class NCXTDBLoader:
    def __init__(self,
                 db,
                 features,
                 cellmask=True,
                 binning=None,
                 norm=None,
                 mask=None,
                 crop=None):
        assert isinstance(features,
                          (list, tuple)), "features must be a list or a tuple"
        self.db = db
        self.features = features
        self.cellmask = cellmask
        self.binning = binning
        self.norm = norm
        self.mask = mask
        self.crop = crop

    @property
    def target_channels(self):
        return 1 + self.cellmask + len(self.features)

    def __len__(self):
        return len(self.db)

    def normalize_lac(self, lac):
        if self.norm is None:
            return lac

        if self.norm == "normalize":
            lac = normalize_nonzero(lac)
        elif self.norm == "equalize":
            lac = equalize_nonzero(lac)
        else:
            try:
                const = float(self.norm)
                lac *= const
            except ValueError:
                msg = f"norm {self.norm} must be a numerical value,'normalize' or 'equalize'"
                raise ValueError(msg)
        return lac

    def feature_selector(self, label, key):
        return feature_selector(
            label, key, self.features, cellmask=self.cellmask)

    def __getitem__(self, index):
        record = self.db[index]
        lac = io.load(record["data"])
        label = io.load(record["annotation"])

        label_sel, key = self.feature_selector(label, record["key"])

        if self.mask or self.crop:
            assert self.mask in [
                None,
                "cellmask",
            ], f"cellmask {self.mask} must be 'cellmask"
            mask, _ = feature_selector(label, record["key"], [], cellmask=True)

            if self.mask:
                lac *= mask

            if self.crop:
                try:
                    pad = float(self.crop)
                except ValueError:
                    msg = f"norm {self.crop} must be a numerical value"
                    raise ValueError(msg)
                lac = crop_to_nonzero(lac, mask, pad=pad)
                label_sel = crop_to_nonzero(label_sel, mask, pad=pad)

        lac_input = self.bin_image(1.0 * self.normalize_lac(lac), mode=False)
        label_sel = self.bin_image(label_sel, mode=True)

        return {
            "input": lac_input.reshape(1, *lac_input.shape),
            "target": label_sel.astype(int),
            "key": key,
        }

    def bin_image(self, image, mode=False):
        if self.binning:
            binfunc = bin_ndarray_mode if mode else bin_ndarray_single
            bin_shape = [s // self.binning for s in image.shape]
            crop = [s * self.binning for s in bin_shape]
            image = binfunc(image[:crop[0], :crop[1], :crop[2]], self.binning)

        return image

    def lac_to_input(self, lac):
        retval = lac.copy()
        retval = self.bin_image(self.normalize_lac(retval), mode=False)
        return retval.reshape(1, *retval.shape)


class NCXTDBCNNLoader(NCXTDBLoader):
    def __init__(
            self,
            db,
            network,
            features,
            cellmask=True,
            binning=None,
            norm=None,
            mask=None,
            crop=None,
    ):
        super().__init__(db, features, cellmask, binning, norm, mask, crop)
        self.network = network

    def __getitem__(self, index):
        sample = NCXTDBLoader.__getitem__(self, index)
        data = sample["input"]
        cell_mask = self.network.apply(data, probability=True)
        sample["input"] = np.stack((data[0], cell_mask[1]), axis=0)
        return sample

    def lac_to_input(self, lac):
        data = NCXTDBLoader.lac_to_input(self, lac)
        cell_mask = self.network.apply(data, probability=True)
        return np.stack((data[0], cell_mask[1]), axis=0)


class ContextLoaderFeatures:
    def __init__(self, seg, folder, features, cellmask=True):
        self.seg = seg
        self.folder = folder
        self.features = features
        self.cellmask = cellmask
        self.sample_paths = [
            folder + name for name in os.listdir(folder) if ".json" in name
        ]

        params = {"working_directory": "d:/2019/SXT_CNN/bin8/", "downscale": 8}
        self.seg.loadstate()

    def __len__(self):
        return len(self.sample_paths)

    def __getitem__(self, index):
        data_path = self.sample_paths[index]

        with open(data_path) as f:
            data = json.load(f)
        lac = io.load("/".join([self.folder, data["name"], data["lac"]]))
        label = io.load("/".join(
            [self.folder, data["name"], data["labelfield"]]))

        label, key = feature_selector(
            label, data["key"], self.features, cellmask=self.cellmask)
        features = self.seg.features(lac)

        data = np.array([lac, *features])

        retval = {"input": data, "target": label.astype(int), "key": key}
        return retval


class FeatureLoader:
    def __init__(self, loader, featurefunc):
        self.loader = loader
        self.featurefunc = featurefunc

    def __len__(self):
        return len(self.loader)

    def __getitem__(self, index):
        data = self.loader[index]
        target = data["target"]
        key = data["key"]

        orig_data = data["input"]
        features = self.featurefunc(orig_data)

        # for feature in features:
        #     ncxt_utils.mrc.MRC_static(feature)

        data_out = np.concatenate((orig_data, features), axis=0)

        retval = {"input": data_out, "target": target, "key": key}
        return retval
