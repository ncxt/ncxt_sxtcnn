import unittest
import ncxt_sxtcnn
import numpy as np
import os
from pathlib import Path


class TestLoad(unittest.TestCase):
    def test_float(self):
        float_data = ncxt_sxtcnn.ncxtamira.read_mrc("data/blob.mrc")
        float_data63 = ncxt_sxtcnn.ncxtamira.loadfloat(
            "data/ver63/template-files/blob.tiff"
        )
        np.testing.assert_array_equal(float_data, float_data63)

    def test_label(self):
        label_data = ncxt_sxtcnn.ncxtamira.read_mrc("data/blob.labels.mrc")
        label_data63, _ = ncxt_sxtcnn.ncxtamira.loadlabel(
            "data/ver63/template-files/blob.labels"
        )
        np.testing.assert_array_equal(label_data, label_data63)

    def test_project63(self):
        float_data = ncxt_sxtcnn.ncxtamira.read_mrc("data/blob.mrc")
        label_data = ncxt_sxtcnn.ncxtamira.read_mrc("data/blob.labels.mrc")
        project = ncxt_sxtcnn.ncxtamira.loadproject("data/ver63/template.hx")
        np.testing.assert_array_equal(float_data, project.lac)
        np.testing.assert_array_equal(label_data, project.labels)

    def test_project2019(self):
        float_data = ncxt_sxtcnn.ncxtamira.read_mrc("data/blob.mrc")
        label_data = ncxt_sxtcnn.ncxtamira.read_mrc("data/blob.labels.mrc")
        project = ncxt_sxtcnn.ncxtamira.loadproject("data/ver2019/template.hx")
        np.testing.assert_array_equal(float_data, project.lac)
        np.testing.assert_array_equal(label_data, project.labels)

    def test_template(self):
        project = ncxt_sxtcnn.ncxtamira.loadproject("data/ver63/template.hx")

        template = ncxt_sxtcnn.ncxtamira.AmiraCell(
            project.lac, project.labels, project.key, name="__temp__"
        )
        template.export("_temp/")

        project_template = ncxt_sxtcnn.ncxtamira.loadproject("_temp/__temp__.hx")

        np.testing.assert_array_equal(project_template.lac, project.lac)
        np.testing.assert_array_equal(project_template.labels, project.labels)

        # cleanup
        os.remove("_temp/__temp__.hx")
        os.remove("_temp/__temp__-files/__temp__.labels")
        os.remove("_temp/__temp__-files/__temp__.rec")
        os.rmdir("_temp/__temp__-files")
        os.rmdir("_temp")


class TestHx(unittest.TestCase):
    def setUp(self):
        self.hx63 = Path("data/ver63/template.hx")

    def test_find_label(self):
        from ncxtamira.project import find_load

        with open(self.hx63, mode="r") as fileobj:
            lines = fileobj.readlines()
            self.assertIsNone(find_load(lines, "dummy"))
            self.assertIsNotNone(find_load(lines, "labels"))

    def test_connection(self):
        from ncxtamira.project import find_load, parse_imagedata

        with open(self.hx63, mode="r") as fileobj:
            lines = fileobj.readlines()
            label_name = Path(find_load(lines, ".labels")).name
            self.assertIsNone(parse_imagedata(lines, "dummy"))
            self.assertIsNotNone(parse_imagedata(lines, label_name))


if __name__ == "__main__":
    unittest.main()
