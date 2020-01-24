import unittest

import ncxt_sxtcnn

from ncxt_sxtcnn.sxtcnn.loaders import MockLoader
from ncxt_sxtcnn.sxtcnn.models import UNet3D
from ncxt_sxtcnn.sxtcnn.processors import RandomBlockProcessor
from ncxt_sxtcnn.sxtcnn.processors import RandomSingleBlockProcessor
from ncxt_sxtcnn.sxtcnn.criteria import CrossEntropyLoss
from ncxt_sxtcnn.sxtcnn.sxtcnn import SXTCNN


class TestCNN(unittest.TestCase):
    def setUp(self):
        self.n_labels = 3
        self.temp_wd = "__tempwd__/"

        self.loader = MockLoader(shape=(3, 4, 5), out_channels=self.n_labels, length=3)
        self.processor = RandomSingleBlockProcessor(block_shape=(16, 16, 16))
        self.criteria = CrossEntropyLoss()

        self.params = {"cfm_step": 2}

        model = UNet3D(self.n_labels)
        seg = SXTCNN(
            self.loader,
            self.processor,
            model,
            self.criteria,
            working_directory=self.temp_wd,
            conf=self.params,
        )
        seg.init_data([0, 1], [2])

    def test_devices(self):
        for device in ["cpu", "cuda"]:
            print(f" === {device} ===")
            model = UNet3D(self.n_labels)
            params = {"device": device}
            seg = SXTCNN(
                self.loader,
                self.processor,
                model,
                self.criteria,
                working_directory=self.temp_wd,
                conf=params,
            )
            seg.init_data([0, 1], [2])
            seg.epoch_step()

    def test_pipe_cuda(self):
        model = UNet3D(self.n_labels)
        params = {"device": "cuda"}
        seg = SXTCNN(
            self.loader,
            self.processor,
            model,
            self.criteria,
            working_directory=self.temp_wd,
            conf=params,
        )
        seg.init_data([0, 1], [2])
        seg.run()
        seg.evaluate_sample(0, self.loader, plot=False)

    def test_pipe_cpu(self):
        model = UNet3D(self.n_labels)
        params = {"device": "cpu"}
        seg = SXTCNN(
            self.loader,
            self.processor,
            model,
            self.criteria,
            working_directory=self.temp_wd,
            conf=params,
        )
        seg.init_data([0, 1], [2])
        seg.run()
        seg.evaluate_sample(0, self.loader, plot=False)

    def test_set_device(self):
        model = UNet3D(self.n_labels)
        params = {"device": "cpu"}
        seg = SXTCNN(
            self.loader,
            self.processor,
            model,
            self.criteria,
            working_directory=self.temp_wd,
            conf=params,
        )
        seg.init_data([0, 1], [2])
        seg.epoch_step()
        seg.set_device("cuda:0")
        seg.epoch_step()


if __name__ == "__main__":
    unittest.main()
