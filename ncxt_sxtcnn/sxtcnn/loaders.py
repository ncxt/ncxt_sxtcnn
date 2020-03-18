import numpy as np
import ncxtamira


class MockLoader:
    def __init__(self, shape, in_channels=1, out_channels=2, length=1):
        self.shape = shape
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        key = {f"material_{n}": n for n in range(self.out_channels)}
        random_lac = np.random.random((self.in_channels, *self.shape))
        random_label = np.random.randint(
            low=0, high=self.out_channels, size=self.shape, dtype="int"
        )

        return {"input": random_lac, "target": random_label.astype(int), "key": key}

    def __call__(self, data):
        retval = data.copy()
        return retval.reshape(1, *retval.shape)


def feature_selector(image, keydict, features, cellmask=False):
    features = features.copy()
    if "*" in features:
        cellmask = True
        features.remove("*")

    keydict = dict((k.lower(), v) for k, v in keydict.items())

    cell_labels = [v for k, v in keydict.items() if "ext" not in k]
    ignore_labels = [v for k, v in keydict.items() if "ignore" in k]

    ignore_mask = np.isin(image, ignore_labels).astype(int)

    retval_key = {"exterior": 0}

    label = (
        np.isin(image, cell_labels).astype(int)
        if cellmask
        else np.zeros(image.shape, dtype=int)
    )
    if cellmask:
        retval_key["cell"] = 1

    for i, keys in enumerate(features):
        index = cellmask + 1 + i
        # print(i, keys, index)
        # make feature iterable
        if not isinstance(keys, (list, tuple)):
            keys = [keys]
        for key in keys:
            try:
                value = keydict[key]
                label[image == value] = index
                retval_key[key] = index
            except KeyError:
                pass

    if np.sum(ignore_mask):
        index = max(retval_key.values()) + 1
        retval_key["ignore"] = index
        label[ignore_mask > 0] = index
    return label, retval_key


class FeatureSelector:
    def __init__(self, key, *features):

        self.key = key.copy()
        self.material_dict = {"void": 0}
        self.cellmask = False
        self.features = []
        for i, feature in enumerate(list(*features)):
            if not isinstance(feature, (list, tuple)):
                feature = [feature]
            if "*" in feature:
                self.cellmask = True
                self.material_dict["cell"] = 1
            else:
                self.features.append(feature)

        for i, feature in enumerate(self.features):
            index = self.cellmask + 1 + i
            for material in feature:
                if material in self.key:
                    self.material_dict[material] = index

    def __call__(self, image):
        # print(self.key, "-->", self.material_dict)

        cell_labels = [v for k, v in self.key.items() if "void" not in k]
        ignore_labels = [v for k, v in self.key.items() if "ignore" in k]
        ignore_mask = np.isin(image, ignore_labels).astype(int)

        retlabel = (
            np.isin(image, cell_labels).astype(int)
            if self.cellmask
            else np.zeros(image.shape, dtype=int)
        )
        for k, v in self.material_dict.items():
            if k not in ["void", "cell"]:
                retlabel[image == self.key[k]] = v

        if np.sum(ignore_mask):
            index = max(self.material_dict.values()) + 1
            self.material_dict["ignore"] = index
            retlabel[ignore_mask > 0] = index

        return retlabel, self.material_dict


class AmiraLoader:
    def __init__(self, files, features):
        self.files = files
        self.features = features

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        data = ncxtamira.project.CellProject(self.files[index])
        lac_input = data.lac
        label_sel, key = FeatureSelector(data.key, self.features)(data.labels)

        return {
            "input": lac_input.reshape(1, *lac_input.shape),
            "target": label_sel.astype(int),
            "key": key,
        }

    def __call__(self, data):
        retval = data.copy()
        return retval.reshape(1, *retval.shape)


class AmiraLoaderx100:
    def __init__(self, files, features):
        self.files = files
        self.features = features
        # todo assert features in CellProject

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        data = ncxtamira.project.CellProject(self.files[index])
        lac_input = data.lac * 100
        label_sel, key = FeatureSelector(data.key, self.features)(data.labels)

        return {
            "input": lac_input.reshape(1, *lac_input.shape),
            "target": label_sel.astype(int),
            "key": key,
        }

    def __call__(self, data):
        retval = data.copy() * 100
        return retval.reshape(1, *retval.shape)


from scipy.ndimage import gaussian_filter


class CascadeAmiraLoader:
    def __init__(self, segmenter, files, features):
        self.files = files
        self.features = features
        self.sigma = 2
        self._segmenter = segmenter
        self._loader = copy.deepcopy(segmenter.loader)
        self._loader.files = files
        self._loader.features = features

    def __len__(self):
        return len(self.files)

    def cascade_input(self, data):
        model_prediction = self._segmenter.model_probability(data)
        for index, _ in enumerate(model_prediction):
            model_prediction[index] = gaussian_filter(
                model_prediction[index], sigma=self.sigma
            )
        model_prediction[0] = data
        return model_prediction

    def __getitem__(self, index):
        sample = self._loader[index]
        sample["input"] = self.cascade_input(sample["input"])
        return sample

    def __call__(self, volume):
        return self.cascade_input(self._loader(volume))


import copy


class OneHotCascadeAmiraLoader:
    def __init__(self, segmenter, files, features):
        self.files = files
        self.features = features
        self.sigma = 2
        self._segmenter = segmenter
        self._loader = copy.deepcopy(segmenter.loader)
        self._loader.files = files
        self._loader.features = features

    def __len__(self):
        return len(self.files)

    def cascade_input(self, data):
        model_prediction = self._segmenter.model_prediction(data)
        n_values = self._segmenter.model.num_classes
        onehot = np.transpose(np.eye(n_values)[model_prediction], (3, 0, 1, 2))
        onehot[0] = data
        return onehot

    def __getitem__(self, index):
        sample = self._loader[index]
        sample["input"] = self.cascade_input(sample["input"])
        return sample

    def __call__(self, volume):
        return self.cascade_input(self._loader(volume))
