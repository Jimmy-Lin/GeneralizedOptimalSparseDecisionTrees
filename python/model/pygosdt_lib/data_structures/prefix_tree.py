# Implementation of set and dictionary
#  - Stores sequential keys using prefix-sharing to achieve lower memeory usage
#  - Efficeintly queries the key-set for shortest-known or longest-known prefix length

# Purpose:
#  The purpose of this datastructure is mainly for efficient querying of whether a task's dependency path has already been pruned
#  Each task in queue is a directive to solve a problem due to it being a dependency (at the time) of another problem
#  In cases where an optimization needs to choose the best of k choices, we issue k tasks as dependencies
#  As the optimal result of each k tasks is known to a higher precision with additional time and computation,
#  Some tasks can be pruned away.
#  We store a set of pruned dependency paths, such that any subpaths of a path in this set is known
#  to be issued from a problem that is no longer needed.
#  Halting evaluation of those problems prevents further exploration of even more subproblems and frees
#  up processes to work on other tasks that are actually relevant to the main optimization problem

# Usage: This datastructure has almost the same interface as a set and dictionary
# shortest_prefix and longest_prefix are additional functionalities
# prefixes = PrefixTree()
# a = 'foo'
# b = 'foobar'
# prefixes[a] = 41
# prefixes[b] = 42
# prefixes.shortest_prefix(b) == 3
# prefixes.longest_prefix(b) == 6

class PrefixTree:
    def __init__(self, minimize=False):
        self.node = PrefixTreeNode()
        self.base = 2
        self.size = 0
        self.minimize = minimize
        self.lock = None
    
    def clear(self):
        self.node = PrefixTreeNode()
        self.size = 0

    def accepts(self, key, value):
        return not self.has(key) or self.get(key) != value

    # Returns True if the key exists in the key-set
    def has(self, key):
        node = self.node
        for element in key:
            if element in node.children:
                node = node.children[element]
            else:
                return False  # First mismatch is found
        # No mismatch found return match based on whether current node is terminal
        return node.value != None

    # Returns the value associated with the key if it exists, otherwise return None
    def get(self, key):
        node = self.node
        for element in key:
            if element in node.children:
                node = node.children[element]
            else:
                return None  # First mismatch is found
        # No mismatch found return match based on whether current node is terminal
        return node.value

    # Associates a value with the key
    def put(self, key, value):
        node = self.node
        for element in key:
            if not element in node.children:
                new_node = PrefixTreeNode(key=element, parent=node)
                node.children[element] = new_node
            node = node.children[element]
        node.value = value
        self.size += 1
        if self.minimize:
            for child in node.children.values():
                child.parent = None
            node.children = {}

    # Remove any association with the key if there is one
    def remove(self, key):
        if len(key) == 0 or self.size == 0:
            return  # This specific implementation excludes empty keys so that we don't trivially prefix match everything
        node = self.node
        for element in key:
            if not element in node.children:
                return # Ignore removal if not found
            else:
                node = node.children[element]
        node.value = None
        self.size -= 1
        # Remove trailing nodes to save on memory
        while node.value == None and node.parent != None and len(node.children) == 0:
            parent = node.parent
            del parent.children[node.key]
            node.parent = None
            node = parent

    # Alias for set interface
    def add(self, key, value=True):
        self.put(key, True)

    # Return length i such that key[:i] is the shortest known prefix in this set.
    # If no prefix is found, 0 is trivially returned
    def shortest_prefix(self, key):
        node = self.node
        i = 0
        for i, element in enumerate(key):
            if node.value != None:
                return i # Early termination for shortest prefix match
            if element in node.children:
                node = node.children[element]
            else:
                return 0 # No more matches available
        return i + 1 if node.value != None else 0 # Either key ends on a known prefix or not

    # Return length i such that key[:i] is the longest known prefix in this set.
    # If no prefix is found, 0 is trivially returned
    def longest_prefix(self, key):
        node = self.node
        imax = 0
        for i, element in enumerate(key):
            if node.value != None:
                imax = i # Mark longest known prefix so far
            if element in node.children:
                node = node.children[element]
            else:
                break  # No more matches available
        return i + 1 if node.value != None else imax # Either a prefix was found or not

    # Operator override for dictionary interface
    def __getitem__(self, key):
        return self.get(key)

    # Operator override for dictionary interface
    def __setitem__(self, key, value):
        self.put(key, value)

    # Operator override for dictionary interface
    def __delitem__(self, key):
        self.remove(key)

    # Operator override for set and dictionary interface
    def __contains__(self, key):
        return self.has(key)
    
    def __len__(self):
        return self.size
    
    # Not a terribly fast implementation, this is just for debugging purposes
    def items(self, node=None):
        if node == None:
            node = self.node
        items = {}
        if node.value != None:
            items[''] = node.value
        if len(node.children) > 0:
            for prefix in node.children:
                for suffix, value in self.items(node=node.children[prefix]).items():
                    if suffix == '':
                        items[(prefix,)] = value
                    else:
                        items[(prefix,) + suffix] = value
        return items

# Internal node structure for building the prefix tree
class PrefixTreeNode:
    def __init__(self, key=None, value=None, parent=None):
        self.key = key
        self.value = value
        self.children = {}
        self.parent = parent
