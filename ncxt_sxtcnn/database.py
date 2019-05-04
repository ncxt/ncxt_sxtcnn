from tinydb import TinyDB, Query
import json
from pathlib import Path
from .read_write_mrc import read_mrc
from .plotters import plot_data
from . import io

def entry_from_file(file):
    file = Path(file)
    with open(file, "r") as f:
        data = json.load(f)
    retval = data

    retval = dict()
    retval["name"] = data["name"]
    retval["data"] = str(file.parent / data["lac"])
    retval["annotation"] = str(file.parent / data["labelfield"])
    retval["key"] = data["key"]
    return retval


class NCXTDB:
    def __init__(self, path):
        self._db = TinyDB(path)

    def add_sample(self, file):
        entry = entry_from_file(file)
        # print(f"Adding  {entry['name']}")

        sample = Query()
        if self.contains(sample.name == entry["name"]):
            raise ValueError(f"Duplicate entry {entry['name']}")

        return self.insert(entry)

    def print(self):
        print(f"Datbase contains {len(self._db)} records \n")
        for item in self._db:
            print(item.doc_id, item["name"])

    def __getattr__(self, name):
        """
        Forward all unknown attribute calls to the TinyDB
        """
        return getattr(self._db, name)

    def __len__(self):
        return len(self._db)

    def __getitem__(self, arg):
        # print(arg)
        if arg < 0 or arg >= self.__len__():
            raise IndexError
        record = self._db.get(doc_id=(arg + 1))
        return record


def plot_record(record):
    lac = io.load(record["data"])
    label = io.load(record["annotation"])
    key = record["key"]
    plot_data(lac, label, record["name"], key)
