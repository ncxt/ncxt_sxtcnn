import _blocks
from _blocks import bin_volume, bin_tensor
from _blocks import upscale_volume, upscale_tensor

from .utils import bin_label, bin_ndarray, bin_tensor
from .volumeblocks import (fuse, fuse_py, random_blocks, random_blocks_label,
                           split, split_label, split_py)