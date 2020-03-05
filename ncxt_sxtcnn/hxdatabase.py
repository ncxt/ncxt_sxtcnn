import pandas as pd
from ncxt_sxtcnn.database import AmiraDatabase


class Database:
    def __init__(self, folder="y:\\Axel\\AmiraProjects\\Database\\"):
        self.folder = folder
        self.db = AmiraDatabase(folder)

    def dataframe(self):
        return self.db.dataframe()

    def __getitem__(self, index):
        return self.db._records[index]._hxpath

    def dataframe_sel(self, *args):
        df = self.dataframe()
        sel = df["sample"] != ""
        for label in args:
            if not isinstance(label, (list, tuple)):
                label = [label]
            sel_label = df["sample"] == ""
            for synonym in label:
                if synonym in df:
                    sel_label = (sel_label) | (df[synonym] != "")
            sel = (sel) & (sel_label)
        return df[sel]

    def filelist(self, *args):
        dfsel = self.dataframe_sel(*args)
        hxfiles = [str(self[i]) for i in dfsel.index]
        return hxfiles

