from pptree import Node, print_tree


class Node:
    def __init__(self, name, parent=None):
        self.name = name
        self.parent = parent
        self.children = []

        if parent:
            self.parent.children.append(self)

    def flatten(self):
        retval = [self.name]
        for child in self.children:
            retval.extend(child.flatten())
        return retval

    def find(self, x):
        if self.name is x:
            return self
        for node in self.children:
            n = node.find(x)
            if n:
                return n
        return None

    @property
    def depth(self):
        if self.parent:
            return self.parent.depth + 1
        return 0


class LabelTree:
    def __init__(self):
        self._root = Node("sample")

        cell = Node("cell", self._root)
        capillary = Node("capillary", self._root)
        buffer = Node("buffer", self._root)

        membrane = Node("membrane", cell)
        nucleus = Node("nucleus", cell)
        mitochondria = Node("mitochondria", cell)
        chloroplast = Node("chloroplast", cell)

        er = Node("endoplasmic reticulum", cell)
        granule = Node("granule", cell)
        golgi = Node("golgi", cell)

        nucleolus = Node("nucleolus", nucleus)
        heterochromatin = Node("heterochromatin", nucleus)
        euchromatin = Node("euchromatin", nucleus)

    def pptree(self):
        print_tree(self._root)

    def flat_children(self, label):
        return self._root.find(label).flatten()

    def extract_materials(self, labels):
        indexdict = dict()

        max_depth = 0

        nodes = sorted(
            [self._root.find(label) for label in labels],
            key=lambda x: (x.depth, x.name),
        )

        for i, node in enumerate(nodes):
            # print(node.depth, node.name)
            for key in node.flatten():
                indexdict[key] = i

        retval = [[] for _ in range(len(labels))]
        for key, val in indexdict.items():
            retval[val].append(key)

        return retval


Organelles = LabelTree()


if __name__ == "__main__":
    # node = Organelles._root.find("nucleus")
    # print(node.name)
    # print(node.flatten())

    # print_tree(Organelles._root)
    # print_tree(node)
    features = Organelles.extract_materials(["cell", "chloroplast", "nucleus"])
