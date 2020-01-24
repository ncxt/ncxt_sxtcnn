import unittest

import ncxt_sxtcnn

from ncxt_sxtcnn.sxtcnn.loaders import AmiraLoaderx100
from ncxt_sxtcnn.sxtcnn.models import UNet3D
from ncxt_sxtcnn.sxtcnn.processors import RandomSingleBlockProcessor
from ncxt_sxtcnn.sxtcnn.criteria import CrossEntropyLoss_DiceLoss
from ncxt_sxtcnn.segmenter import Segmenter
import fnmatch
import os


class TestCNN(unittest.TestCase):
    def setUp(self):
        self.hxfiles = []
        for root, dir, files in os.walk("data/"):
            for file in fnmatch.filter(files, "*.hx"):
                self.hxfiles.append(os.path.join(root, file))

        features = ["*", "nucleus"]
        loader_args = {"files": self.hxfiles, "features": features}
        processor_args = {"block_shape": (16, 16, 16), "n_blocks": 2}
        model_args = {"num_classes": 3}
        criterion_args = {"ignore_index": 2}
        cnn_args = {"batch_size": 10, "maximum_iterations": 10}

        self.seg = Segmenter(
            AmiraLoaderx100,
            RandomSingleBlockProcessor,
            UNet3D,
            CrossEntropyLoss_DiceLoss,
            loader_args=loader_args,
            processor_args=processor_args,
            model_args=model_args,
            criterion_args=criterion_args,
            settings=cnn_args,
        )
        self.seg.folder = "__temp__/"
        self.seg.setup()

    def test_stable_hash_after_run(self):
        path1 = self.seg.jsonpath
        _ = self.seg.kfold_result(2)
        path2 = self.seg.jsonpath
        self.assertEqual(path1, path2)

    def test_kfold0(self):
        path1 = self.seg.jsonpath
        _ = self.seg.kfold_result(0)
        path2 = self.seg.jsonpath
        self.assertEqual(path1, path2)

    def test_from_dict(self):
        seg2 = Segmenter.from_dict(self.seg.export_dict())
        self.assertEqual(seg2.hash, self.seg.hash)


if __name__ == "__main__":
    unittest.main()
