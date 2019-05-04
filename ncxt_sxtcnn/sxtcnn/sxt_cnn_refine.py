import itertools
import os
import random
from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from scipy.signal.windows import triang
from sklearn.metrics import confusion_matrix
from torch import optim
from torch.utils.data import Dataset
from tqdm import trange

from .blocks import (combine, combine_tensor, split_wrapper_md,
                     split_wrapper_multilabel, split_wrapper_tensor, upscale,
                     upscale_dims)
from .cnnutils import features, model_confusion_matrix
from .dataloaders import FeatureLoader
from .datasets import TrainBlocks
from .models import RefUNet3D, UNet3D
from .sxt_cnn import SXT_CNN
from .utils import get_slices

from .datasets import TorchList
from .cnnutils import apply_model


# TODO:
# change calss to eat a trained SXT_CNN as first layer
class SXT_binCNN():
    def __init__(self, loader, params=None):
        self.params = {
            'name': 'SXT_binCNN',
            'downscale': [8, 4],
            'block_shape': (32, 32, 32),
            'gpu': True,
            'batch_size': 4,
            'split': 0.7,
            'sampling': 1.05
        }
        if params is not None:
            self.params.update(params)

        in_channels = 1
        out_channels = 3  #TODO:fix
        start_filts = 4
        depth = 3

        self.models = [
            UNet3D(
                out_channels,
                in_channels=1,
                depth=depth,
                start_filts=start_filts,
                batchnorm=True),
            RefUNet3D(
                out_channels,
                in_channels=1,
                skip_channels=start_filts,
                depth=depth,
                start_filts=start_filts,
                batchnorm=True),
        ]

        if self.params['gpu']:
            for m in self.models:
                m.cuda()

        self.criterion = nn.CrossEntropyLoss(ignore_index=-1)

        self.train_res = dict()
        self.valid_res = dict()
        self.cfm_step = 5
        self.train_cfm = dict()
        self.valid_cfm = dict()
        for i, scale in enumerate(self.params['downscale']):
            self.train_res[i] = [[] for i in range(3)]
            self.valid_res[i] = [[] for i in range(3)]

            self.train_cfm[i] = []
            self.valid_cfm[i] = []

        self.loader = loader
        self.max_data = None
        self.epoch = 0

        self.name = self.params['name']
        self.statedata = ['name', 'epoch', 'train_res', 'valid_res']

    def savestate(self):
        folder = self.params['working_directory']
        state = dict()
        for key in self.statedata:
            state[key] = self.__getattribute__(key)

        np.save(f'{folder}/{self.name}_state.npy', state)
        for i, m in enumerate(self.models):
            torch.save(m.state_dict(),
                       f'{folder}/{self.name}_model{i}_{self.epoch}.weights')

    def loadstate(self):
        folder = self.params['working_directory']
        state = np.load(f'{folder}/{self.name}_state.npy').item()

        for key in self.statedata:
            self.__setattr__(key, state[key])

        for i, m in enumerate(self.models):
            m.load_state_dict(
                torch.load(
                    f'{folder}/{self.name}_model{i}_{self.epoch}.weights'))

    def memoize_data(self, loader, index, max_data):
        scale = self.params['downscale'][index]
        print(f'memoize_data scale {scale}')

        wd = Path(self.params['working_directory'])
        name = f'index{index}'

        train = wd / name / 'train'
        validation = wd / name / 'validation'
        if not os.path.exists(train):
            os.makedirs(train)
        if not os.path.exists(validation):
            os.makedirs(validation)

        block_shape = self.params['block_shape']
        sampling = self.params['sampling']
        block_shape_big = [s * scale for s in block_shape]

        index_train = 0
        index_test = 0
        for i, sample in enumerate(loader):
            data = sample['input']
            labels = sample['target']
            print(f'loader sample: input {data.shape} target {labels.shape}')

            blocks_x = split_wrapper_tensor(
                data, block_shape_big, binning=scale,
                sampling=sampling)['blocks']

            blocks_y = split_wrapper_multilabel(
                labels, block_shape_big, binning=scale,
                sampling=sampling)['blocks']

            n_blocks = len(blocks_x)
            n_split = int(self.params['split'] * n_blocks)

            # seed = hash(path.split('/')[-1]) % 512
            seed = 1
            random_generator = np.random.RandomState(seed)
            idx = random_generator.permutation(n_blocks)
            idx_train = idx[:n_split]
            idx_test = idx[n_split:]

            print(
                f'Saving  {len(idx_train)} train  {len(idx_test)} test blocks')

            for ind in idx_train:
                data_dict = {'x': blocks_x[ind], 'y': blocks_y[ind]}
                np.save(train / f'data{index_train}', data_dict)
                index_train += 1

            for ind in idx_test:
                data_dict = {'x': blocks_x[ind], 'y': blocks_y[ind]}
                np.save(validation / f'data{index_test}', data_dict)
                index_test += 1

            if i + 1 >= max_data:
                return

    def init_data(self, max_data):
        self.max_data = max_data
        self.memoize_data(self.loader, 0, max_data)

    def train(self, index, lr=1e-4, cfm=False):
        scale = self.params['downscale'][index]
        model = self.models[index]
        optimizer = optim.Adam(model.parameters(), lr=1e-4)

        wd = Path(self.params['working_directory'])
        name = f'index{index}'
        folder = str(wd / name / 'train') + '/'

        train_loader = torch.utils.data.DataLoader(
            TrainBlocks(folder), batch_size=self.params['batch_size'])

        model.train()
        batch_loss = []
        cfm_accumulate = np.zeros((model.num_classes, model.num_classes))

        for i_batch, sample_batched in enumerate(train_loader):
            # print(f'Training batch {i_batch}')
            inputs, labels, indecies = sample_batched
            labels = labels.long()
            if self.params['gpu']:
                inputs = inputs.cuda()
                labels = labels.cuda()

            optimizer.zero_grad()
            model_out = model(inputs)
            loss = self.criterion(model_out, labels)
            loss.backward()
            optimizer.step()
            batch_loss.append(loss.item())

            if cfm:
                cfm_accumulate += model_confusion_matrix(
                    model_out, labels, model.num_classes)

        return (np.mean(batch_loss), np.min(batch_loss),
                np.max(batch_loss)), cfm_accumulate

    def validation(self, index, cfm=False):
        scale = self.params['downscale'][index]
        model = self.models[index]

        wd = Path(self.params['working_directory'])
        name = f'index{index}'
        folder = str(wd / name / 'validation') + '/'

        validation_loader = torch.utils.data.DataLoader(
            TrainBlocks(folder), batch_size=self.params['batch_size'])

        model.eval()
        batch_loss = []
        cfm_accumulate = np.zeros((model.num_classes, model.num_classes))

        with torch.no_grad():
            for i_batch, sample_batched in enumerate(validation_loader):
                inputs, labels, indecies = sample_batched
                labels = labels.long()
                if self.params['gpu']:
                    inputs = inputs.cuda()
                    labels = labels.cuda()

                model_out = model(inputs)
                loss = self.criterion(model_out, labels)
                batch_loss.append(loss.item())

                if cfm:
                    cfm_accumulate += model_confusion_matrix(
                        model_out, labels, model.num_classes)

        return (np.mean(batch_loss), np.min(batch_loss),
                np.max(batch_loss)), cfm_accumulate

    def run_scale(self, index, n_epoch=10, base_lr=1e-4):
        scale = self.params['downscale'][index]
        print(f'Training scale[{index}] {scale} for {n_epoch} epochs')

        for i_epoch in trange(n_epoch):
            self.epoch += 1

            calc_cfm = (self.epoch % self.cfm_step) == 0

            trainres, traincfm = self.train(index, lr=base_lr, cfm=calc_cfm)
            validres, validcfm = self.validation(index, cfm=calc_cfm)

            if calc_cfm:
                self.train_cfm[index].append(traincfm)
                self.valid_cfm[index].append(validcfm)

            for i, loss in enumerate(trainres):
                self.train_res[index][i].append(loss)
            for i, loss in enumerate(validres):
                self.valid_res[index][i].append(loss)

            # check train
            stop_crit = 50
            train_mean = self.train_res[index][0]
            train_crit = self.train_res[index][0][-stop_crit:]
            if np.min(train_mean) != np.min(train_crit):
                print(f'best train not in last {stop_crit}')

    def run(self, n_epoch=10, base_lr=1e-4):

        for index, scale in enumerate(self.params['downscale']):
            if index > 0:
                print(f'init_data for scale {scale}')
                self.memoize_data(next_loader, index, self.max_data)

            self.run_scale(index, n_epoch=n_epoch, base_lr=base_lr)

            block_shape = self.params['block_shape']
            batch_size = self.params['batch_size']
            gpu = self.params['gpu']
            model = self.models[index]
            feature_scale = scale

            def featurefunc(image):
                return features(image, model, gpu, feature_scale, block_shape,
                                batch_size)

            next_loader = FeatureLoader(self.loader, featurefunc)

    def plot_train(self):
        n_scales = len(self.params['downscale'])
        f, axis = plt.subplots(2, n_scales, figsize=(8, 8))
        axis = axis.ravel()
        for i, scale in enumerate(self.params['downscale']):
            ax = axis[i]
            train_res = self.train_res[i]
            valid_res = self.valid_res[i]

            x = np.arange(len(train_res[0]))
            ax.plot(x, train_res[0])
            ax.plot(x, valid_res[0])
            ax.fill_between(x, train_res[1], train_res[2], alpha=0.2)
            ax.fill_between(x, valid_res[1], valid_res[2], alpha=0.2)

        colors = [
            '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b',
            '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
        ]

        for i, scale in enumerate(self.params['downscale']):
            ax = axis[n_scales + i]

            train_cfm = self.train_cfm[i]
            valid_cfm = self.valid_cfm[i]

            x = np.arange(len(train_cfm)) * self.cfm_step
            n_labels = train_cfm[0].shape[0]
            print(f'labels {n_labels}')

            for ii in range(n_labels):
                y_prec = [m[ii, ii] / np.sum(m[ii, :]) for m in train_cfm]
                y_rec = [m[ii, ii] / np.sum(m[:, ii]) for m in train_cfm]
                y_dice = [
                    2 * m[ii, ii] / (np.sum(m[:, ii]) + np.sum(m[ii, :]))
                    for m in train_cfm
                ]

                ax.plot(
                    x,
                    y_dice,
                    label=f'Label {ii}',
                    color=colors[ii],
                    linewidth=2)
                ax.plot(x, y_prec, color=colors[ii], linestyle='--')
                ax.plot(x, y_rec, color=colors[ii], linestyle='-.')
            ax.legend()

    def plot_example(self, index=0, sample=0, mode='train'):
        scale = self.params['downscale'][index]
        model = self.models[index]

        wd = Path(self.params['working_directory'])
        name = f'index{index}'
        folder = str(wd / name / mode) + '/'

        loader = torch.utils.data.DataLoader(
            TrainBlocks(folder), batch_size=self.params['batch_size'])

        model.eval()
        with torch.no_grad():
            for i_batch, sample_batched in enumerate(loader):
                if i_batch == sample:
                    inputs, labels, indecies = sample_batched
                    labels = labels.long()
                    if self.params['gpu']:
                        output = torch.softmax(
                            model(inputs.cuda()), dim=1).cpu()
                        features = model.features(inputs.cuda()).cpu()
                        output_labels = torch.argmax(output, dim=1).numpy()

                    else:
                        output = torch.softmax(model(inputs), dim=1)
                        features = model.features(inputs)
                    break

        inputs = inputs.numpy()[0]
        labels = labels.numpy()[0]
        output_label = torch.argmax(output, dim=1).numpy()[0]
        output = output.numpy()[0]
        features = features.numpy()[0]
        print(f'features {features.shape}')
        print(f'model {model.__class__}')

        images = []
        names = []

        # for i in range(inputs.shape[0]):
        for i in range(1):
            images.append(np.concatenate(get_slices(inputs[i, :]), 0))
            names.append(f'input{i}')

        images.append(np.concatenate(get_slices(labels), 0))
        names.append(f'labels')
        images.append(np.concatenate(get_slices(output_label), 0))
        names.append(f'result')

        for i in range(output.shape[0]):
            images.append(np.concatenate(get_slices(output[i, :]), 0))
            names.append(f'output{i}')

        for i in range(features.shape[0]):
            images.append(np.concatenate(get_slices(features[i, :]), 0))
            names.append(f'feature{i}')

        f, axis = plt.subplots(2, len(images) // 2, figsize=(13, 8))
        axis = axis.ravel()
        for i, el in enumerate(zip(axis, images)):
            ax, im = el
            ax.imshow(im)
            ax.set_axis_off()
            ax.set_title(names[i])

    def apply_index(self, image, index, sampling=1.05):
        binning = self.params['downscale'][index]
        model = self.models[index]

        block_shape = self.params['block_shape']
        batch_size = self.params['batch_size']
        is_gpu = self.params['gpu']

        retval = apply_model(image, model, binning, block_shape, batch_size,
                             sampling, is_gpu)

        return np.argmax(retval, axis=0)

    def apply(self, image):
        retval = []
        data_out = image

        for index, scale in enumerate(self.params['downscale']):
            print(f'apply for scale {scale}')
            if index > 0:
                print('init_data for scale {scale}')
                context = featurefunc(image)
                data_out = np.concatenate((image, context), axis=0)

            retval.append(self.apply_index(data_out, index))

            block_shape = self.params['block_shape']
            batch_size = self.params['batch_size']
            gpu = self.params['gpu']
            model = self.models[index]
            feature_scale = scale

            def featurefunc(image):
                return features(image, model, gpu, feature_scale, block_shape,
                                batch_size)

        return retval

    def plot_sample(self, index):
        sample = self.loader[index]

        data = sample['input']
        labels = sample['target']
        nlabels = np.max(labels) + 1
        model_result = self.apply(data)

        imgs = imgs = [
            *get_slices(data[0]),
            *get_slices(labels),
            *get_slices(model_result[0]),
            *get_slices(model_result[1]),
        ]

        cnf_matrix0 = confusion_matrix(labels.ravel(), model_result[0].ravel())
        cnf_matrix1 = confusion_matrix(labels.ravel(), model_result[1].ravel())

        plt.figure(figsize=(13, 13))
        gs1 = gridspec.GridSpec(4, 3)
        gs1.update(left=0.05, right=0.55, wspace=0.05)
        axis = [plt.subplot(g) for i, g in enumerate(gs1)]

        gs2 = gridspec.GridSpec(2, 1)
        gs2.update(left=0.60, right=0.98, hspace=0.05)

        for i, el in enumerate(zip(axis, imgs)):
            ax, im = el
            ax.imshow(im)
            ax.axis('off')

        class_names = [int(x) for x in np.unique(labels)]
        axm1 = plt.subplot(gs2[0])
        plot_confusion_matrix(
            axm1,
            cnf_matrix0,
            classes=class_names,
            title='Normalized(prec) confusion matrix ')
        axm2 = plt.subplot(gs2[1])
        plot_confusion_matrix(
            axm2,
            cnf_matrix1,
            classes=class_names,
            title='Normalized(prec) confusion matrix ')


def plot_confusion_matrix(ax,
                          cm,
                          classes,
                          normalize=True,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.set_title(title)
    # plt.colorbar()
    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(classes, rotation=45)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(
            j,
            i,
            format(cm[i, j], fmt),
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black")

    ax.set_ylabel('True label')
    ax.set_xlabel('Predicted label')
