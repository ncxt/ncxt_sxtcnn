import os
import random
from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import volumeblocks
from scipy.signal.windows import triang
from sklearn.metrics import confusion_matrix
from torch import optim
from torch.utils.data import Dataset
from .utils import rangebar

from .cnnutils import (apply_model, confusion_matrix_metric,
                       model_confusion_matrix, model_features,
                       plot_confusion_matrix)
from .datasets import TorchList, TrainBlocks
from .utils import ensure_dir, get_slices


class SXT_CNN:
    def __init__(self, model, params=None):
        self.params = {
            "name": "SXT_CNN",
            "downscale": 8,
            "block_shape": (32, 32, 32),
            "gpu": True,
            "batch_size": 4,
            "split": 0.7,
            "num_workers": 4,
        }
        if params is not None:
            self.params.update(params)

        self.model = model

        if self.params["gpu"]:
            self.model.cuda()

        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-4)
        self.criterion = nn.CrossEntropyLoss(ignore_index=-1)

        self.epoch = 0
        self.epoch_best = None
        self.max_without_change = 50

        self.train_res = [[] for i in range(3)]
        self.valid_res = [[] for i in range(3)]
        self.cfm_step = 5
        self.train_cfm = []
        self.valid_cfm = []

        self.name = self.params["name"]
        self.folder = self.params["working_directory"]
        self.statedata = [
            "name", "epoch", "epoch_best", "train_res", "valid_res",
            "train_cfm", "valid_cfm"
        ]

    @property
    def file_state(self):
        return f"{self.folder}/{self.name}_state.npy"

    @property
    def file_weights(self):
        return f"{self.folder}/{self.name}_{self.epoch}.weights"

    @property
    def file_weights_best(self):
        return f"{self.folder}/{self.name}_best.weights"

    def savestate(self):
        state = dict()
        for key in self.statedata:
            state[key] = self.__getattribute__(key)

        np.save(self.file_state, state)
        torch.save(self.model.state_dict(), self.file_weights)

    def loadstate(self):
        state = np.load(self.file_state).item()
        for key in self.statedata:
            self.__setattr__(key, state[key])

        self.model.load_state_dict(torch.load(self.file_weights))

    def loadbest(self):
        state = np.load(self.file_state).item()

        for key in self.statedata:
            self.__setattr__(key, state[key])

        self.model.load_state_dict(torch.load(self.file_weights_best))

    def savebest(self):
        if np.argmin(self.valid_res[0]) == self.epoch - 1:
            self.epoch_best = self.epoch
            torch.save(self.model.state_dict(), self.file_weights_best)

    def init_data(self, loader, samples, sampling=1.0, random_seed=1):
        binning = self.params["downscale"]
        block_shape = self.params["block_shape"]
        wd = Path(self.params["working_directory"])

        train = wd / "train"
        validation = wd / "validation"

        for directory in [train, validation]:
            ensure_dir(directory)

        block_shape_big = [s * binning for s in block_shape]
        fileindex_train, fileindex_test = 0, 0
        for index in samples:
            sample = loader[index]

            data = sample["input"]
            labels = sample["target"]

            blocks_x = volumeblocks.split(
                data, block_shape_big, binning=binning, sampling=sampling)

            blocks_y = volumeblocks.split_label(
                labels, block_shape_big, binning=binning, sampling=sampling)

            n_blocks = len(blocks_x)
            n_split = int(self.params["split"] * n_blocks)

            random_idx = np.random.RandomState(random_seed).permutation(
                n_blocks)

            for ind in random_idx[:n_split]:
                data_dict = {"x": blocks_x[ind], "y": blocks_y[ind]}
                np.save(train / f"data{fileindex_train}", data_dict)
                fileindex_train += 1

            for ind in random_idx[n_split:]:
                data_dict = {"x": blocks_x[ind], "y": blocks_y[ind]}
                np.save(validation / f"data{fileindex_test}", data_dict)
                fileindex_test += 1

    def epoch_step(self, cfm=False):
        num_classes = self.model.num_classes

        for step in ['train', 'validation']:
            folder = self.params["working_directory"] + step + "/"
            loader = torch.utils.data.DataLoader(
                TrainBlocks(folder, random_flip=True),
                batch_size=self.params["batch_size"],
                num_workers=self.params["num_workers"])

            if step == 'train':
                self.model.train()
                logs_cfm = self.train_cfm
                logs_loss = self.train_res
            else:
                self.model.eval()
                logs_cfm = self.valid_cfm
                logs_loss = self.valid_res

            num_classes = self.model.num_classes
            batch_loss = []
            cfm_accumulate = np.zeros((num_classes, num_classes))

            for sample_batched in loader:
                inputs, labels = sample_batched
                labels = labels.long()
                if self.params["gpu"]:
                    inputs = inputs.cuda()
                    labels = labels.cuda()

                model_out = self.model(inputs)
                loss = self.criterion(model_out, labels)
                batch_loss.append(loss.item())

                if step == 'train':
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                if cfm:
                    cfm_accumulate += model_confusion_matrix(
                        model_out, labels, num_classes)

            if cfm:
                logs_cfm.append(cfm_accumulate)

            lossres = (np.mean(batch_loss), np.min(batch_loss),
                       np.max(batch_loss))

            for i, loss in enumerate(lossres):
                logs_loss[i].append(loss)

    def run(self, n_epoch=10, learning_rate=1e-4):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = learning_rate

        print(f"Training for {n_epoch} epochs")
        for epoch in rangebar(n_epoch):
            self.epoch += 1
            calc_cfm = (self.epoch % self.cfm_step) == 0
            self.epoch_step(calc_cfm)

            self.savebest()
            if self.epoch - self.epoch_best > self.max_without_change:
                print('no change within max_without_change')
                break

    def plot_train(self):
        x = np.arange(len(self.train_res[0]))
        f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(13, 5))
        ax1.plot(x, self.train_res[0])
        ax1.plot(x, self.valid_res[0])
        ax1.fill_between(x, self.train_res[1], self.train_res[2], alpha=0.2)
        ax1.fill_between(x, self.valid_res[1], self.valid_res[2], alpha=0.2)

        train_cfm = self.train_cfm
        valid_cfm = self.valid_cfm

        x = np.arange(len(train_cfm)) * self.cfm_step
        n_labels = train_cfm[0].shape[0]

        def plot_cfm(ax, cfmlist):
            colors = [
                "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
                "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"
            ]
            for ii in range(n_labels):
                cfm_metric = confusion_matrix_metric
                y_prec = [cfm_metric(m, ii, 'precision') for m in cfmlist]
                y_rec = [cfm_metric(m, ii, 'recall') for m in cfmlist]
                y_dice = [cfm_metric(m, ii, 'dice') for m in cfmlist]
                ax.plot(
                    x,
                    y_dice,
                    label=f"Label {ii} {y_prec[-1]:.2f} / {y_rec[-1]:.2f}",
                    color=colors[ii],
                    linewidth=2,
                )
                ax.plot(x, y_prec, color=colors[ii], linestyle="--")
                ax.plot(x, y_rec, color=colors[ii], linestyle="-.")
            ax.legend()

        plot_cfm(ax2, train_cfm)
        plot_cfm(ax3, valid_cfm)

    def plot_example(self, index=0, mode="train"):
        if mode == "train":
            folder = self.params["working_directory"] + "train/"

        if mode == "valid":
            folder = self.params["working_directory"] + "validation/"

        loader = TrainBlocks(folder, random_flip=False)

        self.model.eval()
        with torch.no_grad():
            inputs, labels = loader[index]
            labels = labels.long()

            inputs = inputs.view(1, *inputs.shape)
            labels = labels.view(1, *labels.shape)

            if self.params["gpu"]:
                output = torch.sigmoid(self.model(inputs.cuda())).cpu()
            else:
                output = torch.sigmoid(self.model(inputs))

        inputs = inputs.numpy()[0]
        labels = labels.numpy()[0]
        output_label = torch.argmax(output, dim=1).numpy()[0]
        output = output.numpy()[0]

        print(f'inputs {inputs.shape}')

        images = []
        names = []

        for i in range(1 + 0 * inputs.shape[0]):
            images.append(np.concatenate(get_slices(inputs[i, :]), 0))
            names.append(f"input{i}")

        images.append(np.concatenate(get_slices(labels), 0))
        names.append(f"labels")
        images.append(np.concatenate(get_slices(output_label), 0))
        names.append(f"result")

        for i in range(output.shape[0]):
            images.append(np.concatenate(get_slices(output[i, :]), 0))
            names.append(f"output{i}")

        f, axis = plt.subplots(1, len(images), figsize=(8, 11))
        axis = axis.ravel()
        for i, el in enumerate(zip(axis, images)):
            ax, im = el
            ax.imshow(im)
            ax.set_axis_off()
            ax.set_title(names[i])

    def apply(self, image, probability=False, sampling=1.05):

        binning = self.params["downscale"]
        block_shape = self.params["block_shape"]
        batch_size = self.params["batch_size"]
        is_gpu = self.params["gpu"]

        retval = apply_model(image, self.model, binning, block_shape,
                             batch_size, sampling, is_gpu)

        if probability:
            return retval
        else:
            return np.argmax(retval, axis=0)

    def evaluate_sample(self, loader, index, plot=False):
        sample = loader[index]

        data = sample["input"]
        labels = sample["target"]
        nlabels = np.max(labels) + 1
        model_result = self.apply(data)

        imgs = imgs = [
            *get_slices(data[0]),
            *get_slices(labels),
            *get_slices(model_result),
        ]
        cnf_matrix = confusion_matrix(labels.ravel(), model_result.ravel())

        if plot == False:
            return cnf_matrix

        plt.figure(figsize=(13, 8))
        gs1 = gridspec.GridSpec(3, 3)
        gs1.update(left=0.05, right=0.55, wspace=0.05)
        axis = [plt.subplot(g) for i, g in enumerate(gs1)]

        gs2 = gridspec.GridSpec(1, 1)
        gs2.update(left=0.60, right=0.98, hspace=0.05)

        for i, el in enumerate(zip(axis, imgs)):
            ax, im = el
            ax.imshow(im)
            ax.axis("off")

        class_names = [int(x) for x in np.unique(labels)]

        ax_cfm = plt.subplot(gs2[0])
        plot_confusion_matrix(
            ax_cfm,
            cnf_matrix,
            classes=class_names,
            title="Normalized(prec) confusion matrix ",
        )

        return cnf_matrix

    def features(self, image):

        binning = self.params["downscale"]
        block_shape = self.params["block_shape"]
        batch_size = self.params["batch_size"]
        is_gpu = self.params["gpu"]

        return model_features(image, self.model, is_gpu, binning, block_shape,
                              batch_size)
