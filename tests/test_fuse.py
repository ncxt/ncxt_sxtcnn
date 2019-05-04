import unittest
import volumeblocks as module
import numpy as np


class TestFuse(unittest.TestCase):
    def test_volume(self):
        image = np.random.random((50, 51, 52)).astype("float32")
        block_shape = (8, 9, 10)

        blocks = module.split(image, block_shape, binning=1)
        image_fused = module.fuse(blocks, image.shape, binning=1)

        np.testing.assert_array_almost_equal(image, image_fused)
        self.assertTrue(True)

    def test_tensor(self):
        for ch in [1, 2, 3]:
            block_shape = (8, 9, 10)
            tensor = np.random.random((ch, 50, 51, 52)).astype("float32")

            blocks = module.split(tensor, block_shape, binning=1)
            tensor_fused = module.fuse(blocks, tensor.shape, binning=1)

            np.testing.assert_array_almost_equal(tensor, tensor_fused)
            self.assertTrue(True)


if __name__ == "__main__":
    unittest.main()
