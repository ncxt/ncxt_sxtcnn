import unittest
import ncxt_sxtcnn as sxtcnn
import numpy as np


class TestBinning(unittest.TestCase):
    def test_volume(self):
        shape = (32, 32, 32)
        for binning in [2, 4, 8]:
            bin_shape = [int(s / binning) for s in shape]
            image = np.random.random(shape).astype("float32")

            bin_ref = sxtcnn.utils.bin_ndarray(image, bin_shape)
            bin_pb = sxtcnn.volumeblocks.bin_volume(image, binning)

            np.testing.assert_array_almost_equal(bin_ref, bin_pb)
            self.assertTrue(True)

    def test_tensor(self):
        for ch in [1, 2, 3]:
            shape = (ch, 32, 32, 32)
            for binning in [2, 4, 8]:
                bin_shape = [int(s / binning) for s in shape]
                bin_shape[0] = ch
                image = np.random.random(shape).astype("float32")

                bin_ref = sxtcnn.utils.bin_ndarray(image, bin_shape)
                bin_pb = sxtcnn.volumeblocks.bin_tensor(image, binning)

                np.testing.assert_array_almost_equal(bin_ref, bin_pb)
                self.assertTrue(True)


class TestUpscale(unittest.TestCase):
    def test_volume(self):
        shape = (32, 32, 32)
        for binning in [2, 4]:
            bin_shape = [int(s / binning) for s in shape]
            image = np.random.random(shape).astype("float32")

            bin_ref = sxtcnn.utils.upscale(image, binning)
            bin_pb = sxtcnn.volumeblocks.upscale_volume(image, binning)

            np.testing.assert_array_almost_equal(bin_ref, bin_pb)
            self.assertTrue(True)

    def test_tensor(self):
        for ch in [1, 2, 3]:
            shape = (ch, 32, 32, 32)
            for binning in [2, 4]:
                bin_shape = [int(s / binning) for s in shape]
                bin_shape[0] = ch
                image = np.random.random(shape).astype("float32")

                bin_ref = sxtcnn.utils.upscale_dims(image, [1, 2, 3], binning)
                bin_pb = sxtcnn.volumeblocks.upscale_tensor(image, binning)

                np.testing.assert_array_almost_equal(bin_ref, bin_pb)
                self.assertTrue(True)


class TestFuse(unittest.TestCase):
    def test_volume(self):
        image = np.random.random((50, 51, 52)).astype("float32")
        block_shape = (8, 9, 10)

        blocks = sxtcnn.volumeblocks.split(image, block_shape, binning=1)
        image_fused = sxtcnn.volumeblocks.fuse(blocks, image.shape, binning=1)

        np.testing.assert_array_almost_equal(image, image_fused)
        self.assertTrue(True)

    def test_tensor(self):
        for ch in [1, 2, 3]:
            block_shape = (8, 9, 10)
            tensor = np.random.random((ch, 50, 51, 52)).astype("float32")

            blocks = sxtcnn.volumeblocks.split(tensor, block_shape, binning=1)
            tensor_fused = sxtcnn.volumeblocks.fuse(blocks, tensor.shape, binning=1)

            np.testing.assert_array_almost_equal(tensor, tensor_fused)
            self.assertTrue(True)


if __name__ == "__main__":
    unittest.main()
