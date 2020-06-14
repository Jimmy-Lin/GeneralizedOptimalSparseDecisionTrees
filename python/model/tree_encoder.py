from math import floor
class TreeEncoder:
    def __init__(self, data, key, value = []):
        if value == []:
            data = list(sorted(set(data), key=key))
        if len(data) == 0:
            self.value = value
        else:
            reference_index = floor(len(data)/2)
            self.reference = data[reference_index]
            self.key = key
            self.left = TreeEncoder(data[0:reference_index], key, value = value + [0])
            self.right = TreeEncoder(data[reference_index+1:], key, value = value + [1])
            self.value = None

        if value == []:
            for d in data:
                print(d, " => ", self.encode(d))

    def depth(self):
        if not self.value is None:
            return 0
        else:
            return max( 1 + self.left.depth(), 1 + self.right.depth())
    
    def width(self):
        if not self.value is None:
            return 1
        else:
            return self.left.width() + self.right.width()

    def encode(self, sample):
        depth = self.depth()
        encoded = self.__encode__(sample)
        while len(encoded) < depth:
            encoded = encoded + [0]

        # print(sample, encoded)
        return encoded

    def __encode__(self, sample):
        if not self.value is None:
            return self.value
        
        # print("Original", sample, self.reference)
        # print("Key-ed", self.key(sample), self.key(self.reference))
        if self.key(sample) < self.key(self.reference):
            return self.left.__encode__(sample)
        else:
            return self.right.__encode__(sample)