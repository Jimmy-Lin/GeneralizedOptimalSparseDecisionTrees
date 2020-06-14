from multiprocessing import Lock
from multiprocessing.managers import DictProxy

class HashTable:
    def __init__(self, table=None):
        if type(table) == DictProxy:
            self.table = table
            self.lock = Lock()
        else:
            self.table = table if table != None else {}
            self.lock = None

    def __getitem__(self, key):
        return self.table[key]

    def get(self, key):
        return self.table.get(key)

    def __setitem__(self, key, value):
        if self.lock == None:
            if self.accepts(key, value):
                self.table[key] = value
        else:
            with self.lock:
                if self.accepts(key, value):
                    self.table[key] = value

    def accepts(self, key, value):
        return not key in self.table or self.table[key] != value

    def __contains__(self, key):
        return key in self.table

    def __delitem__(self, key):
        del self.table[key]

    def __str__(self):
        return str(self.table)

    def __repr__(self):
        return repr(self.table)

    def __len__(self):
        return len(self.table)

    def update(self, updates):
        self.table.update(updates)

    def items(self):
        return self.table.items()

    def keys(self):
        return self.table.keys()

    def values(self):
        return self.table.values()

    def clear(self):
        self.table.clear()
