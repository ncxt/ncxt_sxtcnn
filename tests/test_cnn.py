import unittest

import ncxt_sxtcnn

from ncxt_sxtcnn.sxtcnn.factory import SegFactory
from ncxt_sxtcnn.sxtcnn.dataloaders import NCXTMockLoader
from ncxt_sxtcnn.sxtcnn.models import UNet3D
from ncxt_sxtcnn.sxtcnn.datainitializers import RandomBlockProcessor
from ncxt_sxtcnn.sxtcnn.datainitializers import SingleBlockProcessor
from ncxt_sxtcnn.sxtcnn.sxt_cnn_wrapper import SXT_CNN_WRAPPER


class TestCNN(unittest.TestCase):
    def setUp(self):
        self.n_labels = 3
        self.temp_wd = "__tempwd__/"

        self.loader = NCXTMockLoader(shape=(3, 4, 5), labels=self.n_labels, length=3)
        self.processor = SingleBlockProcessor(block_shape=(16, 16, 16))

        self.params = {"name": f"test_devices", "working_directory": self.temp_wd}

        model = UNet3D(self.n_labels)
        seg = SXT_CNN_WRAPPER(self.loader, model, self.processor, params=self.params)
        seg.init_data(**{"train_idx": [0, 1], "test_idx": [2], "reset": True})

    def test_devices(self):
        for device in ["cpu", "cuda"]:
            print(f" === {device} ===")
            model = UNet3D(self.n_labels)
            params = {
                "name": f"test_devices",
                "device": device,
                "working_directory": self.temp_wd,
            }
            seg = SXT_CNN_WRAPPER(self.loader, model, self.processor, params=params)
            seg.epoch_step()

    def test_pipe_cuda(self):
        model = UNet3D(self.n_labels)
        params = {
            "name": f"test_devices",
            "device": "cuda",
            "working_directory": self.temp_wd,
        }
        seg = SXT_CNN_WRAPPER(self.loader, model, self.processor, params=self.params)
        seg.run()
        seg.evaluate_sample(0, self.loader, plot=False)
        seg.plot_example(0)

    def test_pipe_cpu(self):
        model = UNet3D(self.n_labels)
        params = {
            "name": f"test_devices",
            "device": "cpu",
            "working_directory": self.temp_wd,
        }
        seg = SXT_CNN_WRAPPER(self.loader, model, self.processor, params=self.params)
        seg.run()
        seg.evaluate_sample(0, self.loader, plot=False)
        seg.plot_example(0)

    def test_set_device(self):
        model = UNet3D(self.n_labels)
        params = {
            "name": f"test_devices",
            "device": "cpu",
            "working_directory": self.temp_wd,
        }
        seg = SXT_CNN_WRAPPER(self.loader, model, self.processor, params=self.params)

        seg.epoch_step()
        seg.set_device("cuda:0")
        seg.epoch_step()


if __name__ == "__main__":
    unittest.main()
