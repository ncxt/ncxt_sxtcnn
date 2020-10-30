""" handling of amira .hx projects"""

import re
from logging import warning
from pathlib import Path

import numpy
from numpy.matrixlib.defmatrix import mat
from scipy import ndimage as ndi

from . import LOGGER
from .fuzzylabels import match_key
from .mesh import AmiraMesh
from .organelles import Organelles
from .plotters import plot_data
from .utils import Matcher, ensure_path

COLORS = [
    "0.7 0.8 0.8",
    "0.37 0.61 0.62",
    "1 0 0",
    "1 1 0",
    "1 0 1",
    "0 1 0",
    "0 0 1",
    "0.5 0.5 0",
    "0 1 1",
    "0.5 0 0.5",
    "0.85 0.4 0.4",
    "0 0.5 1",
    "0.5 0.25 1",
]


def labelcolor(i):
    """Label color module len(COLORS) for materials"""
    return COLORS[i % len(COLORS)]


def find_load(lines, tag):
    """ find a dataload in .hx file containing argument 'tag' """
    re_load = re.compile(r"^.+?\$\{\w+\}\/*(.*)\/(.+?)[\"\s].+$")
    is_loader = Matcher(re_load)
    for line in lines:
        if all(x.lower() in line.lower() for x in ["[ load", tag]):
            if is_loader(line):
                groups = re_load.match(line).groups()
                groups = tuple(g for g in groups if len(g))
                return "/".join(groups)

    LOGGER.warning("No matching connection for %s", tag)
    return None


def parse_imagedata(lines, filename):
    """
    Connected density values are of form:
    "labelimage" ImageData connect "floatimage"

    connected data is the last argument of the line
    """

    for line in lines:
        if all(x in line for x in [filename, "ImageData connect"]):
            return line.split()[-1]

    LOGGER.warning("No matching connection for %s", filename)
    return None


def parse_hx(filename, label_tag=".Labels", data_tag=None):
    """ parse label and data field from .hx file """
    LOGGER.info("parse_hx label_tag  %s data_tag  %s", label_tag, data_tag)

    path = Path(filename)
    with open(filename, mode="r") as fileobj:
        lines = fileobj.readlines()

    label_arg = find_load(lines, label_tag)
    label_path = path.parent / label_arg

    LOGGER.info("Found label_arg  %s", str(label_arg))
    LOGGER.info("Found label_path  %s", str(label_path))

    if data_tag:
        data_arg = find_load(lines, data_tag)
        data_path = path.parent / data_arg
    else:
        LOGGER.warning("find data")
        datafile = parse_imagedata(lines, str(label_path.name))
        data_arg = find_load(lines, datafile)
        data_path = path.parent / data_arg

    LOGGER.info("Found data_arg  %s", str(data_arg))
    LOGGER.info("Found data_path  %s", str(data_path))

    return data_path, label_path


def write_label(filename, array, key):
    """ Write byte precision amira mesh file """
    array = array.astype("uint8")
    nz, ny, nx = array.shape

    material_names = ["NA"] * len(key)
    for k, v in key.items():
        material_names[v] = k

    materials = [
        f"{name} {{ Color {labelcolor(i)} }}" for i, name in enumerate(material_names)
    ]

    # bb_arg = " ".join([f"0 {val-1}" for val in array.shape])
    # content_arg = "x".join([str(val) for val in array.shape])
    parameters = [
        f'Content "{nx}x{ny}x{nz} byte, uniform coordinates"',
        f"BoundingBox 0 {nx-1} 0 {ny-1} 0 {nz-1}",
        f'CoordType "uniform"',
    ]

    ensure_path(filename)
    with open(filename, mode="wb") as fileobj:
        fileobj.write(b"# AmiraMesh BINARY-LITTLE-ENDIAN 2.1\n")
        fileobj.write(str.encode(f"define Lattice {nx} {ny} {nz}\n"))

        fileobj.write(b"Parameters {\n")
        fileobj.write(b"    Materials {\n")
        for material in materials:
            fileobj.write(str.encode(material + ",\n"))
        fileobj.write(b"    }\n")
        for parline in parameters:
            fileobj.write(str.encode(parline + ",\n"))
        fileobj.write(b"}\n")
        fileobj.write(b"Lattice { byte Labels } @1\n")
        fileobj.write(b"# Data section follows\n")
        fileobj.write(b"@1\n")
        fileobj.write(array.tobytes())


class AmiraProject:
    """Minimal amira project with:
        a data image (float)
        a segmentation image (int)
        a material key
    """

    def __init__(self, hxpath, sanitize=True):
        """load .hx amira project

        Arguments:
            hxpath {string} -- Path to .hx project file
        
        Keyword Arguments:
            sanitize {bool} -- sanitize material keys on load (default: {True})
        """

        self.hxpath = Path(hxpath)
        self.sanitize = sanitize
        self.folder = self.hxpath.parent
        self.stem = self.hxpath.stem

        self.lac_file, self.labelfile = parse_hx(hxpath)

        self._lac = None
        self._labels = None
        self._key = None

    # @classmethod
    # def from_data(cls, lac,label, key):
    #     return cls(name, date.today().year - birthYear)

    # def save(self, filepath):
    #     """Save Amira project to path

    #     Arguments:
    #         filepath {string/Path} -- Path to .hx file
    #     """
    #     path = Path(filepath)
    #     assert path.suffix == ".hx", "Save projects as .hx files!"
    #     folder = path.parent
    #     stem = path.stem
    #     # template = AmiraTemplate(self.lac, self.labels, self.key, name=stem)
    #     template.export(folder)

    @property
    def lac(self):
        """ lazy loading float image """
        if self._lac is not None:
            return self._lac

        self._lac = AmiraMesh(self.lac_file).arr
        return self._lac

    @property
    def labels(self):
        """ lazy loading label image """
        if self._labels is not None and self._key is not None:
            return self._labels

        self.loadlabelfile()
        return self._labels

    @property
    def key(self):
        """ lazy loading material key """

        if self._key is not None:
            return self._key

        self._key = AmiraMesh(self.labelfile, headeronly=True).key
        return self._key

    def loadlabelfile(self):
        """ Load labels and key from labelfile"""

        mesh = AmiraMesh(self.labelfile).loadmesh()
        self._key = mesh.key
        self._labels = mesh.arr

    def preview(self):
        """plot a preview of the project"""
        plot_data(self.lac, self.labels, self.stem, self.key)


class AmiraCell:
    """
    Class for containing the cell annotation
    """

    def __init__(self, lac, labels, key, name=None, sanitize=False):
        """ 
        Arguments:
            lac {ndarrays} -- ndarray containing the LAC
            labels {ndarrays} -- ndarray containing the labels
            key {dict} -- dictionary mapping {'material':integer}

        Keyword Arguments:
            name {string} -- sample name (default: {None})
            sanitize {bool} -- Material sanitation. If true, matches the keys in 'self.key' to a list of organelles using Levenshtein distance (default: {False})
        """

        self.name = name
        self._lac = lac
        self._labels = labels
        self._key = key
        self._sanitize = sanitize

        # file paths for lazy loading
        self.hxpath = None
        self.lac_file = None
        self.labelfile = None

        if self._key is not None and self._sanitize:
            self._key = match_key(self._key)

    @classmethod
    def from_hx(cls, hxpath, data_tag=None, **kwargs):
        """Constructor for loading hx files

        Arguments:
            hxpath {string} -- path to the hx file

        Returns:
            [AmiraCell] -- AmiraCell object with lazyloading properties
        """
        hxpath = Path(hxpath)
        retval = cls(None, None, None, name=hxpath.stem, **kwargs)
        retval.hxpath = hxpath
        retval.lac_file, retval.labelfile = parse_hx(hxpath, data_tag=data_tag)
        return retval

    @property
    def lac(self):
        """ lazy loading float image stored in self._lac 

        Returns:
            [ndarray] -- LAC
        """
        if self._lac is not None:
            return self._lac

        LOGGER.info("    Loading lac")
        self._lac = AmiraMesh(self.lac_file).arr
        return self._lac

    @property
    def labels(self):
        """ lazy loading label image stored in self._labels

        Returns:
            [ndarray] -- labels
        """
        if self._labels is not None and self._key is not None:
            return self._labels

        LOGGER.info("    Loading labels")
        self._loadlabelfile()
        return self._labels

    @property
    def key(self):
        """ lazy loading material key stored in self._key

        Returns:
            [dict] -- dictionary of structure {'material':integer}
        """

        if self._key is None:
            LOGGER.info("    Loading key")
            self._key = AmiraMesh(self.labelfile, headeronly=True).key

        if self._sanitize:
            self._key = match_key(self._key)

        return self._key

    def _loadlabelfile(self):
        """ Load labels and key from labelfile"""

        mesh = AmiraMesh(self.labelfile).loadmesh()
        self._key = mesh.key
        self._labels = mesh.arr

    def preview(self):
        """plot a preview of the project"""
        plot_data(self.lac, self.labels, self.name, self.key)

    def select(self, organelle):
        """Return a mask with chosen organelle and all its children

        Arguments:
            organelle {string} -- organelle

        Returns:
            [ndarray] -- Boolean mask of organelle
        """
        if not self._sanitize:
            warning(
                "run 'self.sanitize()'  to ensure that the material"
                "names aare correct"
            )

        return self.bitwise_or(Organelles.flat_children(organelle))

    def bitwise_or(self, keys):
        """Return a mask with chosen organelle and all its children

        Arguments:
            keys {list} -- list of materials

        Returns:
            [ndarray] -- Boolean mask of bitwise_or of keys
        """

        mask = self.labels * 0

        existing_keys = [el for el in keys if self.key.get(el, False)]
        for material in existing_keys:
            mask = numpy.bitwise_or(mask, self.labels == self.key[material])
        return mask

    def sanitize(self):
        """
            Tries to sanitize the material keys with the 
            list of organelles using Levenshtein distance
        """
        _ = self.key
        self._key = match_key(self._key)
        self._sanitize = True

    @property
    def _lacname(self):
        """ name of lac data """
        return f"{self.name}.rec"

    @property
    def _labelname(self):
        """ name of label data """
        return f"{self.name}.labels"

    def _loadline(self, name):
        """return a load line for file "name" for the .hx header"""
        return f'[ load ${{SCRIPTDIR}}/{self.name}-files/{name} ] setLabel "{name}"\n'

    def _write_project(self, filepath):
        """ write project .hx file """
        ensure_path(filepath)
        with open(filepath, mode="wb") as fileobj:
            fileobj.write(b"# Amira Project 630\n")
            fileobj.write(b"# Amira\n")
            fileobj.write(b"# Generated by Amira 6.3.0\n")

            fileobj.write(b"\n")
            fileobj.write(b"# Create viewers\n")
            fileobj.write(b"viewer setVertical 0\n")
            fileobj.write(b"\n")
            fileobj.write(b"viewer 0 setTransparencyType 5\n")
            fileobj.write(b"viewer 0 setAutoRedraw 0\n")
            fileobj.write(b"viewer 0 show\n")
            fileobj.write(b"mainWindow show\n")
            fileobj.write(b"set hideNewModules 0\n")
            fileobj.write(str.encode(self._loadline(self._lacname)))
            fileobj.write(str.encode(self._loadline(self._labelname)))
            fileobj.write(
                str.encode(f'"{self._labelname}" ImageData connect "{self._lacname}"\n')
            )

    def export(self, folder, name=None):
        """ export template project to folder

        Arguments:
            folder {string} -- Export folder

        Keyword Arguments:
            name {string} -- Project name, defaults to self.name (default: {None})
        """

        folder = Path(folder)
        if name:
            self.name = name
        assert self.name, "Name must be specified"

        self._write_project(folder / f"{self.name}.hx")

        write_rec(folder / f"{self.name}-files" / self._lacname, self.lac)
        write_label(
            folder / f"{self.name}-files" / self._labelname, self.labels, self.key
        )

    def zoom(self, zoom):
        zoom_lac = ndi.zoom(self.lac, zoom)
        zoom_labels = ndi.zoom(self.labels, zoom, order=0)

        return AmiraCell(zoom_lac, zoom_labels, self.key)

    @property
    def shape(self):
        assert self.lac.shape == self.labels.shape
        return self.lac.shape


def write_rec(filename, array):
    """ Write floating precision amira mesh file """
    array = array.astype("float32")
    nz, ny, nx = array.shape

    # bb_arg = " ".join([f"0 {val-1}" for val in array.shape])
    # content_arg = "x".join([str(val) for val in array.shape])
    parameters = [
        f'Content "{nx}x{ny}x{nz} float, uniform coordinates"',
        f"BoundingBox 0 {nx-1} 0 {ny-1} 0 {nz-1}",
        f'CoordType "uniform"',
    ]

    ensure_path(filename)
    with open(filename, mode="wb") as fileobj:
        fileobj.write(b"# AmiraMesh BINARY-LITTLE-ENDIAN 2.1\n")
        fileobj.write(str.encode(f"define Lattice {nx} {ny} {nz}\n"))

        fileobj.write(b"Parameters {\n")
        for parline in parameters:
            fileobj.write(str.encode(parline + ",\n"))
        fileobj.write(b"}\n")

        fileobj.write(b"Lattice { float Data } @1\n")
        fileobj.write(b"\n")
        fileobj.write(b"# Data section follows\n")
        fileobj.write(b"@1\n")
        fileobj.write(array.tobytes())

