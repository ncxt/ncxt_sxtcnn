import numpy as np
import os
from functools import reduce
from math import ceil
import torch
from tqdm import trange, tnrange
from scipy.ndimage.filters import maximum_filter


def isnotebook():
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True  # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False  # Probably standard Python interpreter


def rangebar(n):
    rangefunc = tnrange if isnotebook() else trange
    return rangefunc(n)


def ensure_path(file_path):
    '''Ensure the folder exists for file path
    
    Arguments:
        file_path {string} -- full path to file
    '''

    ensure_dir(os.path.dirname(file_path))


def ensure_dir(directory):
    '''Ensure the folder exists 
    
    Arguments:
        file_path {string} -- full path to file
    '''
    if not os.path.exists(directory):
        os.makedirs(directory)


# from https://stackoverflow.com/questions/8090229/resize-with-averaging-or-rebin-a-numpy-2d-array
def bin_ndarray(ndarray, new_shape, operation='mean'):
    """
    Bins an ndarray in all axes based on the target shape, by summing or
        averaging.

    Number of output dimensions must match number of input dimensions and 
        new axes must divide old ones.

    Example
    -------
    >>> m = np.arange(0,100,1).reshape((10,10))
    >>> n = bin_ndarray(m, new_shape=(5,5), operation='sum')
    >>> print(n)

    [[ 22  30  38  46  54]
     [102 110 118 126 134]
     [182 190 198 206 214]
     [262 270 278 286 294]
     [342 350 358 366 374]]

    """
    operation = operation.lower()
    if not operation in ['sum', 'mean']:
        raise ValueError("Operation not supported.")
    if ndarray.ndim != len(new_shape):
        raise ValueError("Shape mismatch: {} -> {}".format(
            ndarray.shape, new_shape))
    compression_pairs = [(d, c // d) for d, c in zip(new_shape, ndarray.shape)]
    flattened = [l for p in compression_pairs for l in p]
    ndarray = ndarray.reshape(flattened)
    for i in range(len(new_shape)):
        op = getattr(ndarray, operation)
        ndarray = op(-1 * (i + 1))
    return ndarray


def get_slices(image):
    s0, s1, s2 = [int(s / 2) for s in image.shape]
    return image[s0, :, :], image[:, s1, :], image[:, :, s2]


def crop_to_tensor_shape(image, ref):
    l_images = image.shape[-1]
    l_ref = ref.shape[-1]
    l_crop = (int)((l_images - l_ref) / 2)
    #     print(f'crop l {l_crop}')
    return image[:, :, l_crop:-l_crop, l_crop:-l_crop, l_crop:-l_crop]


def pad_to_shape(image, shape):
    ''' pad image to give shape '''
    pad = [a - b for a, b in zip(shape, image.shape)]
    return np.pad(image, ((0, pad[0]), (0, pad[1]), (0, pad[2])), 'constant')


def pad_to_div(image, div=4):
    '''pad image woth zeroes si the size is divisibale by div    '''
    div_shape = [div * ceil(x / div) for x in image.shape]
    return pad_to_shape(image, div_shape)


def labels2prob(image):
    n_labels = np.max(image) + 1
    array_label = np.zeros((n_labels, *image.shape))
    for label_index in range(n_labels):
        array_label[label_index, :] = image == label_index
    return array_label


def prob2label_numpy(x):
    x = np.array(x)
    label = np.zeros(x.shape[-3:])
    for n in range(x.shape[0]):
        ind_lab = x[n, :] >= np.max(x, 0)
        label[ind_lab] = n
    return label


def prob2label(x):
    if isinstance(x, np.ndarray):
        return prob2label_numpy(x)
    if isinstance(x, torch.Tensor):
        np_x = x.numpy()
        return prob2label_numpy(np_x)
    raise TypeError('argument should be a np.ndarray or torch.Tensor')


def crop_to_shape(image, shape):
    return image[:shape[0], :shape[1], :shape[2]]


def window(shape, func, **kwargs):
    vs = [func(l, **kwargs) for l in shape]
    return reduce(np.multiply, np.ix_(*vs))


def bin_ndarray_single(ndarray, binning):
    bin_shape = [s // binning for s in ndarray.shape]
    return bin_ndarray(ndarray, bin_shape)


def bin_ndarray_mode(ndarray, binning):
    prob = labels2prob(ndarray)
    bin_shape = [s // binning for s in prob.shape]
    bin_shape[0] = prob.shape[0]
    bin_prob = bin_ndarray(prob, bin_shape)

    return prob2label_numpy(bin_prob)


def getCropLim(image, pad=64, th=0.01):
    # print(image.shape)
    num_el_0 = maximum_filter(np.sum(np.sum(image, 1), 1), 2 * pad + 1)
    num_el_1 = maximum_filter(np.sum(np.sum(image, 0), 1), 2 * pad + 1)
    num_el_2 = maximum_filter(np.sum(np.sum(image, 0), 0), 2 * pad + 1)

    prof_list = [num_el_0, num_el_1, num_el_2]
    count_lim = [th * np.max(x) for x in prof_list]

    clims = [(np.min(np.where(p > c)), np.max(np.where(p > c)))
             for p, c in zip(prof_list, count_lim)]

    return clims


def crop_to_nonzero(img, label, pad=1):
    clim = getCropLim(label, pad=pad)
    return img[clim[0][0]:clim[0][1], clim[1][0]:clim[1][1], clim[2][0]:clim[2]
               [1]]
