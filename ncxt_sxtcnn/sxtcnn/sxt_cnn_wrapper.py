import os
import random
from pathlib import Path
import logging
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

from .cnnutils import (
    apply_model,
    confusion_matrix_metric,
    model_confusion_matrix,
    model_features,
    plot_confusion_matrix,
)
from .datasets import TorchList, TrainBlocks
from .utils import ensure_dir, get_slices, rangebar


class SXT_CNN_WRAPPER:
    def __init__(self, loader, model, processor, params=None):
        self.params = {
            "device": "cuda:0",
            "parallel": False,
            "batch_size": 4,
            "num_workers": 4,
            "ignore": -1,
        }
        if params is not None:
            self.params.update(params)

        self.loader = loader
        self.model = model
        self.processor = processor

        self.num_classes = self.model.num_classes

        if self.params["parallel"]:
            print(f"Eanbling paralell")
            self.model = torch.nn.DataParallel(self.model)

        self.set_device(self.params["device"])
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.params["ignore"])

        self.epoch = 0
        self.epoch_best = None
        self.max_without_change = 10000

        self.train_idx = None
        self.test_idx = None

        self.train_res = [[] for i in range(3)]
        self.valid_res = [[] for i in range(3)]
        self.cfm_step = 5
        self.train_cfm = []
        self.valid_cfm = []

        try:
            self.folder = self.params["working_directory"]
        except KeyError:
            self.folder = None
            logging.warning(
                """
            Working_directory not set
            Member variable 'folder' needs to point to the model directory
            containing the state and the weights
             """
            )

        try:
            self.name = self.params["name"]
        except KeyError:
            self.name = None
            logging.warning(
                """
            Name is not not set
            Model cannot be saved without a name
             """
            )

        self.statedata = [
            "name",
            "epoch",
            "epoch_best",
            "train_res",
            "valid_res",
            "train_cfm",
            "valid_cfm",
        ]

    def set_device(self, device):
        self.device = device
        self.model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-4)

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name

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

    def init_data(self, **kwargs):
        self.train_idx = kwargs["train_idx"]
        self.test_idx = kwargs["test_idx"]

        self.processor.init_data(
            self.loader, working_directory=self.params["working_directory"], **kwargs
        )
        return

    def epoch_step(self, cfm=False):
        for step in ["train", "validation"]:
            folder = self.params["working_directory"] + step + "/"
            loader = torch.utils.data.DataLoader(
                TrainBlocks(folder, random_flip=True),
                batch_size=self.params["batch_size"],
                num_workers=self.params["num_workers"],
            )

            if step == "train":
                self.model.train()
                logs_cfm = self.train_cfm
                logs_loss = self.train_res
            else:
                self.model.eval()
                logs_cfm = self.valid_cfm
                logs_loss = self.valid_res

            batch_loss = []
            cfm_accumulate = np.zeros((self.num_classes, self.num_classes))

            for sample_batched in loader:
                inputs, labels = sample_batched
                labels = labels.long()

                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                # print("Outside: input size", inputs.size())
                model_out = self.model(inputs)
                loss = self.criterion(model_out, labels)
                batch_loss.append(loss.item())

                if step == "train":
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                if cfm:
                    cfm_accumulate += model_confusion_matrix(
                        model_out, labels, self.num_classes
                    )

            if cfm:
                logs_cfm.append(cfm_accumulate)

            lossres = (np.mean(batch_loss), np.min(batch_loss), np.max(batch_loss))

            for i, loss in enumerate(lossres):
                logs_loss[i].append(loss)

    def run(self, n_epoch=10, learning_rate=1e-4):
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = learning_rate

        print(f"Training for {n_epoch} epochs")
        for epoch in rangebar(n_epoch):
            self.epoch += 1
            calc_cfm = (self.epoch % self.cfm_step) == 0
            self.epoch_step(calc_cfm)

            self.savebest()
            if self.epoch - self.epoch_best > self.max_without_change:
                print("no change within max_without_change")
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
                "#1f77b4",
                "#ff7f0e",
                "#2ca02c",
                "#d62728",
                "#9467bd",
                "#8c564b",
                "#e377c2",
                "#7f7f7f",
                "#bcbd22",
                "#17becf",
            ]
            for ii in range(n_labels):
                cfm_metric = confusion_matrix_metric
                y_prec = [cfm_metric(m, ii, "precision") for m in cfmlist]
                y_rec = [cfm_metric(m, ii, "recall") for m in cfmlist]
                y_dice = [cfm_metric(m, ii, "dice") for m in cfmlist]
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

            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            output = torch.sigmoid(self.model(inputs))

        inputs = inputs.cpu().numpy()[0]
        labels = labels.cpu().numpy()[0]
        output_label = torch.argmax(output, dim=1).cpu().numpy()[0]
        output = output.cpu().numpy()[0]

        images = []
        names = []
        clims = []

        for i in range(inputs.shape[0]):
            image = np.concatenate(get_slices(inputs[i, :]), 0)
            images.append(image)
            names.append(f"input{i}")
            clims.append((np.min(image), np.max(image)))

        images.append(np.concatenate(get_slices(labels), 0))
        names.append(f"labels")
        clims.append((0, self.params["ignore"]))

        images.append(np.concatenate(get_slices(output_label), 0))
        names.append(f"result")
        clims.append((0, self.params["ignore"]))

        for i in range(output.shape[0]):
            images.append(np.concatenate(get_slices(output[i, :]), 0))
            names.append(f"output{i}")
            clims.append((0, 1))

        f, axis = plt.subplots(1, len(images), figsize=(8, 11))
        axis = axis.ravel()
        for i, el in enumerate(zip(axis, images)):
            ax, im = el
            ax.imshow(im, clim=clims[i])
            ax.set_axis_off()
            ax.set_title(names[i])

    def apply(self, image, probability=False, sampling=1.05):
        assert self.processor is not None, "apply needs a defined processor"

        processor_output = self.processor.forward(image)

        eval_loader = torch.utils.data.DataLoader(
            TorchList(processor_output), batch_size=self.params["batch_size"]
        )

        self.model.eval()
        model_res = []
        with torch.no_grad():
            for sample_batched in eval_loader:

                sample_batched = sample_batched.to(self.device)
                output = torch.softmax(self.model(sample_batched), dim=1)
                model_res.append(output.cpu().numpy())

        output = np.concatenate(model_res, axis=0)
        # print(f'model_res {output.shape}')

        retval = self.processor.backward(output)
        # print(f'retval {retval.shape}')
        if probability:
            return retval
        else:
            return np.argmax(retval, axis=0)

    def features(self, image):
        assert self.processor is not None, "apply needs a defined processor"

        processor_output = self.processor.forward(image)

        eval_loader = torch.utils.data.DataLoader(
            TorchList(processor_output), batch_size=self.params["batch_size"]
        )

        self.model.eval()
        model_res = []
        with torch.no_grad():
            for sample_batched in eval_loader:
                if self.params["gpu"]:
                    output = self.model.features(sample_batched.cuda()).cpu()
                else:
                    output = self.model.features(sample_batched)
                model_res.append(output.numpy())

        output = np.concatenate(model_res, axis=0)
        # print(f'model_res {output.shape}')

        retval = self.processor.backward(output)
        # print(f'retval {retval.shape}')
        return retval

    def evaluate_sample(self, loader, index, plot=False):
        # print(f'Loading sample')
        sample = loader[index]
        data = sample["input"]
        labels = sample["target"]
        nlabels = np.max(labels) + 1
        # print(f'Applying segmentation')
        model_result = self.apply(data)

        ignore_mask = labels == self.params["ignore"]
        if np.sum(ignore_mask):
            print(f"model {model_result.shape}, target {labels.shape}")
            print(f"Label contains ignore mask {ignore_mask.shape}")

            model_result[ignore_mask] = labels[ignore_mask]

        # print(f'CFM')

        imgs = imgs = [
            *get_slices(data[0]),
            *get_slices(labels),
            *get_slices(model_result),
        ]
        cnf_matrix = _confusion_matrix(
            labels, model_result, labels=[i for i in range(nlabels)]
        )

        # print(f'Plotting')

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
        metrics = ["precision", "recall", "dice"]
        # TODO: fix index
        res = [confusion_matrix_metric(cnf_matrix, 1, m) for m in metrics]
        msg = [f"{m}: {val:.3f}" for m, val in zip(metrics, res)]
        plt.suptitle(msg)

        return cnf_matrix

    def __call__(self, image, **kwargs):
        return self.apply(self.loader.lac_to_input(image, **kwargs))


def _confusion_matrix(a, b, labels):
    # print(f'labels {labels}')
    n_labels = len(labels)
    cfm = np.zeros((n_labels, n_labels))
    for i in labels:
        for j in labels:
            sela = a == i
            selb = b == j
            cfm[i, j] = int(np.sum(sela * selb))
    return cfm
