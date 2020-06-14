from itertools import combinations
from random import sample, shuffle

from model.pygosdt_lib.data_structures.vector import Vector

# Class for storing and performing approximate neighbourhood queries on bitvectors
#  - neighbourhood radius is defined using Hamming distance
# Usage:
# # store sets of bit vectors in the index
# # query for subsets of vectors that are within a maximum hamming distance
# index = SimilarityIndex(distance=1, dimensions=5, tables=5)
# a = Vector('11111')
# b = Vector('01111')
# index.add(b)
# b in index.neighbours(a)
# index.remove(b)
# not b in index.neighbours(a)

class FullIndex:
    def __init__(self):
        print('fullindex')
        self.keys = set()
        self.initialized = False  # Defer table generation until necessary

    # Adds a key (vector) into the the index
    def add(self, key):
        self.keys.add(key)

    # Removes a key (vector) from the the index
    def remove(self, key):
        self.keys.remove(key)

    # Returns a subset of the vectors that are within a distance from key (vector)
    def neighbours(self, key):
        neighbours = self.keys
        return neighbours

    # Override for x in set operator, returns whether a key (vector) is a member of the index
    def __contains__(self, key):
        return key in self.keys

    # Override for str(x) in set operator, displays the set of vectors in this index
    def __str__(self):
        return str(self.keys)

    def __len__(self):
        return len(self.keys)

class SimilarityIndex:
    # Constructor Arguments:
    #  distance = maximum hamming distance to be considered a neighbour
    #  dimensions = the number of dimensions (elements) in each of the bitvectors
    #  tables = the number of hash tables used to catch nearest neighbours

    # The number of tables should be within [1, K], where K = (dimensions choose distance)
    # As the number of tables increase, the false negative rate of neighbour queries decrease
    # More information on the approximation bounds and design are here:
    # https://en.wikipedia.org/wiki/Locality-sensitive_hashing#Bit_sampling_for_Hamming_distance

    def __init__(self, distance=1, dimensions=1,  tables=1):
        if tables < 1:
            raise Exception("SimilarityIndexError: tables {} must be greater than or equal to {}".format(tables, 1))
        if dimensions < 1:
            raise Exception("SimilarityIndexError: dimensions {} must be greater than or equal to {}".format(dimensions, 1))
        if dimensions < distance:
            raise Exception("SimilarityIndexError: distance {} must be less than or equal to dimension {}".format(distance, dimensions))
        self.dimensions = dimensions
        self.distance = distance
        self.table_count = tables
        self.keys = set()
        self.size = 0
        self.initialized = False # Defer table generation until necessary
        

    # (Private) Creates a bitvector of ones with free_dimensions bits set to 0
    # Any vector would have free_dimensions bits set to 0 so that variatiions in those bits are normalized to 0
    def __normalizer__(self, free_dimensions):
        normalizer = Vector([ int(not j in free_dimensions) for j in range(self.dimensions) ])
        return normalizer

    def __build_normalizers__(self, dimensions, degrees_of_freedom, tables):
        normalizers = set()
        while len(normalizers) < tables:
            free_dimensions = tuple(sample(range(dimensions), degrees_of_freedom))
            normalizers.add(free_dimensions)
        return normalizers
        
    # Creates tuples of (vector, dictionary) pairs where the vector indicates which bits are hashing for the respective table
    def __generate_tables__(self, degrees_of_freedom, tables):
        return tuple(
            (self.__normalizer__(combo), {})
            for combo in self.__build_normalizers__(self.dimensions, degrees_of_freedom, tables)
        )

    def initialize(self):
        if self.initialized == False:
            print('Initializing Similarity Index')
            self.tables = self.__generate_tables__(self.distance, self.table_count)
            self.initialized = True
            print('Initialized Similarity Index')


    # Adds a key (vector) into the the index
    def add(self, key):
        if self.initialized == False:
            self.initialize()
        if len(key) != self.dimensions:
            raise Exception("SimilarityIndexError: len(key) {} must be equal to dimension {}".format(len(key), self.dimensions))
        for normalizer, table in self.tables:
            normal_key = key & normalizer
            if not normal_key in table:
                table[normal_key] = set()
            table[normal_key].add(key)
        self.size += 1
        self.keys.add(key)

    # Removes a key (vector) from the the index
    def remove(self, key):
        if self.initialized == False:
            self.initialize()
        if len(key) != self.dimensions:
            raise Exception("SimilarityIndexError: len(key) {} must be equal to dimension {}".format(len(key), self.dimensions))
        for normalizer, table in self.tables:
            normal_key = key & normalizer
            if normal_key in table and key in table[normal_key]:
                table[normal_key].remove(key)
        self.size -= 1
        self.keys.remove(key)

    # Returns a subset of the vectors that are within a distance from key (vector)
    def neighbours(self, key):
        if self.initialized == False:
            self.initialize()
        if len(key) != self.dimensions:
            raise Exception("SimilarityIndexError: len(key) {} must be equal to dimension {}".format(len(key), self.dimensions))
        neighbours = set.union(set(), *(table[key & normalizer] for normalizer, table in self.tables if ((key & normalizer) in table)))
        return neighbours

    # Override for x in set operator, returns whether a key (vector) is a member of the index
    def __contains__(self, key):
        if self.initialized == False:
            self.initialize()
        if len(key) != self.dimensions:
            raise Exception("SimilarityIndexError: len(key) {} must be equal to dimension {}".format(len(key), self.dimensions))
        return key in self.keys

    # Override for str(x) in set operator, displays the set of vectors in this index
    def __str__(self):
        if self.initialized == False:
            self.initialize()
        return str(self.keys)

    def __len__(self):
        return self.size
