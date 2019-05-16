import numpy as np
import torch
from .. import volumeblocks
from scipy.signal.windows import triang
from sklearn.metrics import confusion_matrix
import itertools
from .datasets import TorchList
import matplotlib.pyplot as plt


def plot_confusion_matrix(ax,
                          cm,
                          classes,
                          normalize=True,
                          title="Confusion matrix",
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
      Normalization can be applied by setting `normalize=True`.
    """

    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    ax.imshow(cm, interpolation="nearest", cmap=cmap)
    ax.set_title(title)
    # plt.colorbar()
    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(classes, rotation=45)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(classes)

    fmt = ".2f" if normalize else "d"
    thresh = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(
            j,
            i,
            format(cm[i, j], fmt),
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black",
        )

    ax.set_ylabel("True label")
    ax.set_xlabel("Predicted label")


def confusion_matrix_metric(cfm, index, mode):
    eps = 1e-6

    if mode == 'recall':
        return cfm[index, index] / (eps + np.sum(cfm[index, :]))
    if mode == 'precision':
        return cfm[index, index] / (eps + np.sum(cfm[:, index]))
    if mode == 'dice':
        return 2 * cfm[index, index] / (
            eps + (np.sum(cfm[:, index])) + np.sum(cfm[index, :]))

    raise ValueError("mode must be precision, recall or dice")


def model_confusion_matrix(model_out, labels, num_classes):
    output_labels = torch.argmax(model_out, dim=1)

    if output_labels.is_cuda:
        output_labels = output_labels.cpu()
    if labels.is_cuda:
        labels = labels.cpu()

    cfm = confusion_matrix(
        labels.numpy().ravel(),
        output_labels.numpy().ravel(),
        labels=np.arange(num_classes))
    return cfm


def model_features(image, model, gpu, binning, block_shape, batch_size):
    print(f'running: features:')
    print(f'   image {image.shape}')
    print(f'   model {model.__class__}')
    print(f'   model in{model.in_channels}')
    print(f'   binning{binning} block_shape {block_shape}')

    block_shape_big = [s * binning for s in block_shape]
    sampling = 1.05
    lac_blocks = volumeblocks.split(
        image, block_shape_big, binning=binning, sampling=sampling)

    n_blocks = len(lac_blocks)

    print(
        f"  {n_blocks} blocks , shape {lac_blocks[0].shape} dtype {lac_blocks[0].dtype}"
    )

    eval_loader = torch.utils.data.DataLoader(
        TorchList(lac_blocks), batch_size=batch_size)

    model.eval()
    model_res = []
    with torch.no_grad():
        for sample_batched in eval_loader:
            if gpu:
                output = model.features(sample_batched.cuda()).cpu()
            else:
                output = model.features(sample_batched)
            model_res.append(output.numpy())

    output = np.concatenate(model_res, axis=0)

    return volumeblocks.fuse(
        output, image.shape, binning, sampling, windowfunc=triang)


def apply_model(image, model, binning, block_shape, batch_size, sampling, gpu):
    block_shape_big = [s * binning for s in block_shape]

    lac_blocks = volumeblocks.split(
        image, block_shape_big, binning=binning, sampling=sampling)

    eval_loader = torch.utils.data.DataLoader(
        TorchList(lac_blocks), batch_size=batch_size)

    model.eval()
    model_res = []
    with torch.no_grad():
        for sample_batched in eval_loader:
            if gpu:
                output = torch.softmax(
                    model(sample_batched.cuda()), dim=1).cpu()
            else:
                output = torch.softmax(model(sample_batched), dim=1)
            model_res.append(output.numpy())

    output = np.concatenate(model_res, axis=0)
    retval_shape = [model.num_classes] + list(image.shape)[-3:]

    return volumeblocks.fuse(
        output,
        retval_shape,
        binning=binning,
        sampling=sampling,
        windowfunc=triang)
