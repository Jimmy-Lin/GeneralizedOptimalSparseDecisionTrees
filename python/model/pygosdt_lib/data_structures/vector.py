from gmpy2 import mpz, popcount, bit_mask
from types import GeneratorType

# Wrapper class around MPZ objects made for fixed-length bitvectors
# The wrapper serves the following purposes:
#  - Fixes bug where multiple representation of the same vector introduces false negatives in equality and breaks other comparisons
#  - Adds additional support to Python operators by operator overload to make code look simpler
#  - Bridges mismatch between MPZ indexing and Python iterable indexing
#  - Argument checking on operations

# Basic Usage:
# x = Vector('10101011')
# y = Vector([1,0,1,0,1,0,1,1])
# x ^ y == Vector.zeros(8)

class Vector:
    # Creates a vector of element repeated n-times
    def repeat(element, length, base=2):
        return Vector([element] * length, base=base)

    # Creates a vector of ones repeated n-times
    def ones(length):
        # return Vector.repeat(1, length, base=2)
        return Vector(bit_mask(length), length=length, base=2)

    # Creates a vector of zeros repeated n-times
    def zeros(length):
        return Vector(bit_mask(length) ^ bit_mask(length), length=length, base=2)

    # Creates a vector based on data which is either 
    def __init__(self, data, length=None, base=2):
        if type(data) == type(mpz()):
            self.data = data
            self.length = length
        elif type(data) == str:
            # self.data = mpz(data[::-1], base)
            bitstring = data
            self.data = mpz(bitstring, base)
            self.length = len(bitstring)
        else:
            # if isinstance(data, GeneratorType):
            #     data = tuple(data)
            # self.data = mpz(''.join(str(datum) for datum in data[::-1]), base)
            bitstring = ''.join(repr(datum) for datum in data)
            self.data = mpz(bitstring, base)
            self.length = len(bitstring)
        if self.data < 0:
            # self.data = mpz(self.__str__()[::-1], base) # Workaround if needed
            raise Exception("VectorError: Representation must be non-negative, got {}".format(self.data))
        self.base = base
        self.i = 0

    # Creates a new vector that must have the same length and base
    def generate(self, data):
        return Vector(data, length=self.length, base=self.base)

    # Counts the number of ones in the bitvector
    def count(self):
        return popcount(self.data)

    # Overrides x[i] indexing (Note that assignment isn't supported due to immutability)
    def __getitem__(self, index):
        if index < 0 or index > self.length:
            raise Exception("VectorError: Index {} must be in interval [{}, {}]".format(index, 0, self.length))
        effective_index = self.length - 1 - index
        return int(self.data.bit_test(effective_index))

    # Overrides ~x operator and keeps leading bits as zeros to avoid breaking equality
    def __invert__(self):
        return self.generate(self.data ^ bit_mask(self.length))

    # Overrides x & y operator
    def __and__(self, vector):
        if self.length != vector.length:
            raise Exception("VectorError: Length Mismatch {} and {}".format(self.length, vector.length))
        return self.generate(self.data & vector.data)

    # Overrides x | y operator
    def __or__(self, vector):
        if self.length != vector.length:
            raise Exception("VectorError: Length Mismatch {} and {}".format(self.length, vector.length))
        return self.generate(self.data | vector.data)

    # Overrides x ^ y operator
    def __xor__(self, vector):
        if self.length != vector.length:
            raise Exception("VectorError: Length Mismatch {} and {}".format(self.length, vector.length))
        return self.generate(self.data ^ vector.data)

    # Overrides x < y operator
    def __lt__(self, vector):
        if self.length != vector.length:
            raise Exception("VectorError: Length Mismatch {} and {}".format(self.length, vector.length))
        return self.data < vector.data

    # Overrides x <= y operator
    def __le__(self, vector):
        if self.length != vector.length:
            raise Exception("VectorError: Length Mismatch {} and {}".format(self.length, vector.length))
        return self.data <= vector.data

    # Overrides x > y operator
    def __gt__(self, vector):
        if self.length != vector.length:
            raise Exception("VectorError: Length Mismatch {} and {}".format(self.length, vector.length))
        return self.data > vector.data

    # Overrides x >= y operator
    def __ge__(self, vector):
        if self.length != vector.length:
            raise Exception("VectorError: Length Mismatch {} and {}".format(self.length, vector.length))
        return self.data >= vector.data

    # Overrides x == y operator
    def __eq__(self, vector):
        if type(vector) != Vector:
            return False
        if self.length != vector.length:
            raise Exception("VectorError: Length Mismatch {} and {}".format(self.length, vector.length))
        return self.data == vector.data

    # Overrides x != y operator
    def __ne__(self, vector):
        if type(vector) != Vector:
            return True
        return self.data != vector.data

    # Overrides x * y operator (Assumes base = 2)
    def __mul__(self, vector):
        if self.length != vector.length:
            raise Exception("VectorError: Length Mismatch {} and {}".format(self.length, vector.length))
        return (self & vector).count()

    # Overrides len(x)
    def __len__(self):
        return self.length

    # Overrides str(x)
    def __str__(self):
        return ''.join( str(self[i]) for i in range(self.length) )
    
    # Overrides for e in x: structure
    def __iter__(self):
        return ( self[i] for i in range(self.length) )
    
    # Overrides hashing (vectors can be used as hash keys)
    def __hash__(self):
        return self.data.__hash__()

