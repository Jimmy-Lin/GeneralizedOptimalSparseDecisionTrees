from model.pygosdt_lib.data_structures.prefix_tree import PrefixTree
from model.pygosdt_lib.data_structures.task import Task

class PathCluster:
    def __init__(self):
        self.cluster = PrefixTree()
        self.size = 0

    def add(self, task):
        if type(task) == Task:
            self.cluster.add(task.path)
            self.size = len(self.cluster)
        else:
            self.size += 1

    def remove(self, task):
        if type(task) == Task:
            self.cluster.remove(task.path)
            self.size = len(self.cluster)
        else:
            self.size = max(self.size - 1, 0)

    def proximity(self, task):
        if type(task) == Task:
            # 0 = No match
            # 1 = Perfect match
            return self.cluster.shortest_prefix(task.path) / max(len(task.path), 1)
        else:
            return 0

    def __len__(self):
        return self.size
