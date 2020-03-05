"""
container for amira files
"""
import fnmatch
import os
from pathlib import Path

import matplotlib.pyplot as plt
import ncxtamira
import numpy as np
import pandas as pd
from ncxtamira import AmiraProject, CellProject
from .plotters import make_overlay, get_middle_slices
from .plotters import COLORS
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec
from .sxtcnn.utils import rangebar, ensure_dir


class Record:
    def __init__(self, hxpath):
        self._hxpath = Path(hxpath)

        p = ncxtamira.CellProject(hxpath)
        self._key = p.key

        self.project = self._hxpath.parent.stem
        self.sample = self._hxpath.stem

    def datarow(self):
        retval = dict()
        retval.update({k: v for k, v in vars(self).items() if not k.startswith("_")})
        retval.update({k: int(v) for k, v in self._key.items()})
        return retval


def record_overlaydata(project, datapath, reset=False):
    exists = os.path.exists(datapath)
    # print(datapath, exists)
    if exists:
        stamp_snap = os.path.getmtime(datapath)
        for f in [project.hxpath, project.lac_file, project.labelfile]:
            if os.path.getmtime(f) > stamp_snap:
                print(f"newer stamp for {f}")
                reset = True
                os.remove(datapath)
                break
    # print(f"exists: {exists} reset {reset}")

    if not exists or reset:
        # print(f"Generate data for {datapath}")
        slices_lac = get_middle_slices(project.lac)
        slices_label = get_middle_slices(project.labels)
        void_idx = [v for k, v in project.key.items() if "void" in k]
        overlay_kwargs = {
            "void": void_idx,
            "saturation": 0.6,
            "blend": 0.6,
            "colors": COLORS,
        }

        overlay_images = [
            make_overlay(a, b, **overlay_kwargs)
            for a, b, in zip(slices_lac, slices_label)
        ]
        ensure_dir(datapath)

        np.save(datapath, [overlay_images, slices_label, project.key])

    return np.load(datapath)


class AmiraDatabase:
    def __init__(self, folder=None):
        self._records = []
        self.folder = folder
        print(folder)
        if folder:
            self.add_folder(folder)

    def add(self, path):
        try:
            self._records.append(Record(path))
        except:
            print(f"Cannot add record at {path}")
            raise

    def add_folder(self, path):
        for root, dir, files in os.walk(path):
            for file in fnmatch.filter(files, "*.hx"):
                self.add(os.path.join(root, file))

    def dataframe(self):
        df = pd.DataFrame([record.datarow() for record in self._records])
        cols = list(df)
        cols.insert(0, cols.pop(cols.index("sample")))
        cols.insert(0, cols.pop(cols.index("project")))
        df = df.reindex(columns=cols)
        df = df.replace(np.nan, "", regex=True)
        return df

    def __getitem__(self, index):
        return CellProject(self._records[index]._hxpath)

    def __len__(self):
        return len(self._records)

    def generate_preview(self, raw=False, reset=False):
        for index in rangebar(len(self._records)):
            hxpath = self._records[index]._hxpath
            project = AmiraProject(hxpath) if raw else CellProject(hxpath)
            datapath = Path(self.folder) / "__snapshots__"
            datapath = (
                datapath / (project.stem + "_raw") if raw else datapath / project.stem
            ).with_suffix(".npy")
            _ = record_overlaydata(project, datapath, reset)

    def preview(self, index, raw=False):
        hxpath = self._records[index]._hxpath
        project = AmiraProject(hxpath) if raw else CellProject(hxpath)
        datapath = Path(self.folder) / "__snapshots__"
        datapath = (
            datapath / (project.stem + "_raw") if raw else datapath / project.stem
        ).with_suffix(".npy")

        overlay_images, slices_label, key = record_overlaydata(project, datapath)

        _ = plt.figure(figsize=(13, 5))
        axes = [plt.subplot(gsi) for gsi in gridspec.GridSpec(1, 4)]
        for axis, image in zip(axes, overlay_images):
            axis.imshow(image)

        labels = list(key.keys())

        im_colors = set()
        for image in slices_label:
            im_colors |= set(np.unique(image))
        void_idx = [v for k, v in key.items() if "void" in k]
        im_colors = [i for i in im_colors if i not in void_idx]

        legend_elements = [
            patches.Patch(
                facecolor=COLORS[key[label] % len(COLORS)], edgecolor=None, label=label
            )
            for i, label in enumerate(labels)
            if key[label] in im_colors
        ]

        for axis in axes:
            axis.set_xticks([])
            axis.set_yticks([])
            for spine in axis.spines.values():
                # pass
                spine.set_visible(False)

        axes[3].legend(
            handles=legend_elements,
            loc="upper right",
            bbox_to_anchor=(0.9, 0.9),
            prop={"size": 12},
        )

        plt.subplots_adjust(
            left=0.01, right=0.99, hspace=0.05, wspace=0.01, top=0.9, bottom=0.05
        )

        plt.suptitle(hxpath.stem)

