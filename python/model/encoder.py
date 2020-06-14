import operator

from math import ceil, log
from numpy import array, argsort, hstack, dtype, issubdtype, isnan, nan, int64, int32, float64, float32
from pandas import isnull
from sortedcontainers import SortedList, SortedSet, SortedDict
from sklearn.ensemble import RandomForestClassifier

# Converts arbitrary datasets to binary values
class Encoder:
    def __is_numeric__(value):
        return issubdtype(dtype(type(value)), dtype(int)) or issubdtype(dtype(type(value)), dtype(float))

    def __is_undefined__(value):
        return isnull(value)

    def __predicate__(value):
        if Encoder.__is_numeric__(value) and not Encoder.__is_undefined__(value):
            return operator.ge
        elif Encoder.__is_numeric__(value) and Encoder.__is_undefined__(value):
            return operator.eq
        else:
            return operator.eq

    def __init__(self, data, header=None, mode="complete", target=None):
        self.base = 10
        if len(data.shape) <= 1:
            data = array([ [data[i]] for i in range(data.shape[0]) ])
        (n, m) = data.shape

        if target is None:
            target = [i for i in range(n)]
        else:
            target = target.tolist()

        self.mode = mode
        self.values = [ SortedDict() for j in range(m)] # Track the uniquely occuring values for each column
        self.types = ['Numerical' for j in range(m)]
        self.optional = [True for j in range(m)]
        self.cardinalities = [ None for j in range(m)]
        self.defaults = [None for j in range(m)]
        self.magnitudes = [None for j in range(m)]
        self.radii = [None for j in range(m)]
        self.radial_values = [ SortedDict() for j in range(m)] # Track the uniquely occuring values for each column
        self.names = header

        if self.mode == "none":
            self.headers = header
            return

        if self.mode == "tree":
            from python.model.tree_encoder import TreeEncoder
            (n,m) = data.shape
            clf = RandomForestClassifier(n_estimators=10, max_features=None, random_state=0)
            clf.fit(data, array([ [i] for i in target ]))  
            clf.feature_importances_
            importance_index = argsort(clf.feature_importances_)[::-1]
            def key(sample):
                # print(tuple(sample[i] for i in importance_index))
                return tuple(sample[i] for i in importance_index)
            self.tree = TreeEncoder([ tuple(data[i,:]) for i in range(n) ], key)
            # print("Unique Samples: {}, Tree Width: {} ".format(len(set(tuple(data[i,:]) for i in range(n))), self.tree.width()))
            self.headers = [ "path[{}]".format(i) for i in range(self.tree.depth()) ]
            return

        for j in range(m):
            for i in range(n):
                value = data[i,j]
                if isnull(value) or value == '':
                    self.optional[j] = True
                elif Encoder.__is_numeric__(value):
                    self.types[j] = 'Numerical'
                    if not value in self.values[j]:
                        self.values[j][value] = set()
                    self.values[j][value].add(target[i])

                    # Radial Analysis
                    mantissa = float(value)
                    magnitude = 0
                    if mantissa != 0:
                        while round(mantissa) != mantissa:
                            mantissa = mantissa * self.base
                            magnitude += 1
                        while round(mantissa / self.base) == (mantissa / self.base):
                            mantissa = mantissa / self.base
                            magnitude -= 1

                    if self.magnitudes[j] is None:
                        self.magnitudes[j] = magnitude
                    else:
                        self.magnitudes[j] = max(self.magnitudes[j], magnitude)
                else:
                    self.types[j] = 'Categorical'
                    if not value in self.values[j]:
                        self.values[j][value] = set()
                    self.values[j][value].add(target[i])

            self.cardinalities[j] = len(self.values[j])
            if self.types[j] == 'Numerical' and len(self.values[j]) > 1:

                while len(set( int(value * pow(self.base, self.magnitudes[j])) for value in self.values[j].keys())) == len(set( int(value * pow(self.base, self.magnitudes[j]-1)) for value in self.values[j].keys())):
                    self.magnitudes[j] -= 1

                mantissa = int(max(self.values[j].keys()) * pow(self.base, self.magnitudes[j]))
                self.radii[j] = ceil(log(mantissa, self.base))
                values = list(sorted( value * pow(self.base, self.magnitudes[j]) for value in self.values[j].keys()))

                for k in range(self.radii[j]):
                    radial_values = SortedSet( int(value / pow(self.base, k)) % self.base for value in values)
                    if len(radial_values) > 1:
                        self.radial_values[j][k] = radial_values      

        self.encoders = [None for j in range(m)]

        for j in range(m):
            try:
                if self.cardinalities[j] <= 1 and self.optional == False:
                    self.types[j] = 'Redundant'
                elif self.cardinalities[j] <= 2:
                    self.types[j] = 'Binary'
                elif self.cardinalities[j] <= 5 and all(type(value) == type(0) for value in self.values[j]):
                    self.types[j] = 'Categorical'
                elif self.mode == "radix" and sum(len(values) for radial_index, values in self.radial_values[j].items()) < self.cardinalities[j]:
                    self.types[j] = 'Radial'

                if self.types[j] == 'Redundant':
                    self.encoders[j] = None
                elif self.types[j] == 'Binary':
                    if self.optional[j]:
                        self.encoders[j] = [
                            { 'relation': operator.eq, 'reference': 0, 'value': 0 },
                            { 'relation': operator.eq, 'reference': 1, 'value': 1 },
                        ]
                    else:
                        self.defaults[j] = min(self.values[j])
                        self.encoders[j] = [
                            { 'relation': operator.eq, 'reference': 1, 'value': 1 },
                        ]
                elif self.types[j] == 'Categorical':
                    if self.optional[j]:
                        self.encoders[j] = [ { 'relation': operator.eq, 'reference': value, 'value': value } for value in self.values[j] ]
                    else:
                        self.defaults[j] = min(self.values[j])
                        self.encoders[j] = [ { 'relation': operator.eq, 'reference': value, 'value': value } for value in list(self.values[j].keys())[1:] ]
                elif self.types[j] == 'Radial':
                    if self.optional[j]:
                        self.defaults[j] = min(self.values[j])
                    self.encoders[j] = []
                    for k in range(self.radii[j]):
                        if k in self.radial_values[j] and len(self.radial_values[j][k]) > 1:
                            for radial_value in self.radial_values[j][k]:
                                self.encoders[j].append({ 'relation': operator.ge, 'reference': radial_value, 'value': radial_value , "radial_index": k - self.magnitudes[j] })
                elif self.types[j] == "Numerical":
                    feature_values = list(self.values[j].keys())
                    base_value = feature_values[0]
                    references = [base_value]
                    for value in feature_values[1:]:
                        reference = (base_value + value) / 2.0
                        union = self.values[j][base_value].union(self.values[j][value])
                        if self.mode == "bucketize" and len(union) <= 1:
                            references.append(None)
                        else:
                            references.append(reference)
                        base_value = value
                    if self.optional[j]:
                        self.encoders[j] = [ { 'relation': operator.ge, 'reference': references[k], 'value': feature_values[k] } for k in range(0, len(self.values[j])) if not references[k] is None ]
                    else:
                        self.defaults[j] = min(feature_values)
                        self.encoders[j] = [ { 'relation': operator.ge, 'reference': references[k], 'value': feature_values[k] } for k in range(1, len(self.values[j])) if not references[k] is None ]
            except Exception as e:
                print("Exception({}) at Column {}".format(e, j))
                print("Cardinalities[j] = {}".format(self.cardinalities[j]))
                print("Optional[j] = {}".format(self.optional[j]))
                print("Type[j] = {}".format(self.types[j]))
                print("Values[j] = {}".format(self.values[j]))
                print([type(values) for values in self.values[j]])
                raise e

        # discrete = 0
        # continuous = 0
        # for j in range(m):
        #     if self.types[j] == 'Numerical':
        #         continuous += 1
        #     else:
        #         discrete += 1
        #     if not header is None:
        #         print("Index: {}, Name: {}, Type: {}, Cardinality: {}, Encoders: {}, Optional: {}, Radius: {}, Magnitude: {}".format(
        #             j, header[j], self.types[j], self.cardinalities[j], len(self.encoders[j]), self.optional[j], self.radii[j], self.magnitudes[j]))
        #     else:
        #         print("Index: {}, Name: {}, Type: {}, Cardinality: {}, Encoders: {}, Optional: {}, Radius: {}, Magnitude: {}".format(
        #             j, 'Unknown', self.types[j], self.cardinalities[j], len(self.encoders[j]), self.optional[j], self.radii[j], self.magnitudes[j]))
        # print("Number of Discrete Columns: {}, Number of Continuous Columns: {}, Total Cardinality: {}\n".format(discrete, continuous, sum(self.cardinalities)))

        offset = 0
        self.headers = []
        for j, encoders in enumerate(self.encoders):
            if encoders != None:
                if self.types[j] == "Radial":
                    self.headers += [ 
                        '{}{}{}'.format(
                            'x[i,{}][{}]'.format(j, encoder['radial_index']) if header is None else '{}[{}]'.format(header[j], encoder['radial_index']),
                            '>=' if encoder['relation'] == operator.ge else '==',
                            encoder['reference']
                        ) for encoder in encoders ]
                else:
                    self.headers += [ 
                        '{}{}{}'.format(
                            'x[i,{}]'.format(j) if header is None else header[j],
                            '>=' if encoder['relation'] == operator.ge else '==',
                            encoder['reference']
                        ) for encoder in encoders ]
        # print(self.headers)
    
    def encode(self, data):
        if self.mode == "none":
            return data

        if len(data.shape) <= 1:
            data = array([[data[i]] for i in range(data.shape[0])])
        (n, m) = data.shape
        encoded = []
        for i in range(n):
            if self.mode == "tree":
                # print(data[i,:], "=>", self.tree.encode(data[i,:]))
                encoded.append(self.tree.encode(data[i,:]))
                continue
            row = []                
            for j in range(m):
                encoders = self.encoders[j]
                if encoders == None:
                    continue
                cells = []
                for encoder in encoders:
                    if isnan(data[i, j]):
                        cell = 0
                    elif "radial_index" in encoder:
                        try:
                            value = int(data[i, j] * pow(self.base, -encoder["radial_index"])) % self.base
                        except:
                            print(data[i, j], pow(self.base, -encoder["radial_index"]))
                        cell = int(encoder['relation'](value, encoder['reference']))
                    else:
                        cell = int(encoder['relation'](data[i, j], encoder['reference']))
                    cells.append(cell)
                row += cells
            encoded.append(row)
        return array(encoded)

    def decode(self, data):
        if len(data.shape) <= 1:
            data = array([[data[i]] for i in range(data.shape[0])])
        (n, m) = data.shape
        decoded = []
        for i in range(n):
            offset = 0
            row = []
            for j, encoders in enumerate(self.encoders):
                # print(data[i, offset:offset+len(encoders)])
                if any(data[i, offset+k] == 1 for k in len(encoders)):
                    row.append(max(encoder['value'] for k, encoder in enumerate(encoders) if data[i, offset+k] == 1))
                else:
                    row.append(self.defaults[j])
                offset += len(encoders)
            decoded.append(row)
        return array(decoded)

    def decode_leaves(self, tree):
        if tree['leaf'] == True:
            # del tree['leaf']
            total = sum(weight for _, weight in tree['distribution'].items())
            distribution = {}
            for encoded_label, weight in tree['distribution'].items():
                bits = [int(c) for c in str(encoded_label)]
                bits.reverse() # Must reverse due to endianess
                decoded_label = self.defaults[0]
                encoders = self.encoders[0]
                if any(bits[k] == 1 for k in range(len(encoders))):
                    decoded_label = max(encoder['value'] for k, encoder in enumerate(encoders) if bits[k] == 1)
                if type(decoded_label) in { int, int64, int32 }:
                    distribution[int(decoded_label)] = weight / total
                elif type(decoded_label) in { float, float64, float32 }:
                    distribution[float(decoded_label)] = weight / total
                else:
                    distribution[str(decoded_label)] = weight / total
            tree['distribution'] = distribution
            if not self.names is None:
                tree['name'] = self.names[0]
        elif tree['leaf'] == False:
            for value, subtree in tree['branches'].items():
                self.decode_leaves(subtree)
        return tree

    def decode_branches(self, tree):
        if 'leaf' in tree and tree['leaf'] == False:
            for value, subtree in tree['branches'].items():
                self.decode_branches(subtree)
            # del tree['leaf']
            encoded_feature_index = tree['feature_index']
            offset = 0
            tree['branches'][str(False)] = tree['branches'][str(int(False))]
            tree['branches'][str(True)] = tree['branches'][str(int(True))]
            del tree['branches'][str(int(False))]
            del tree['branches'][str(int(True))]
            for j, encoders in enumerate(self.encoders):
                if encoded_feature_index < offset + len(encoders):
                    tree['feature_index'] = j
                    encoder = encoders[encoded_feature_index - offset]
                    tree['reference'] = encoder['reference']
                    if encoder['relation'] == operator.ge:
                        tree['relation'] = '>='
                    elif encoder['relation'] == operator.eq:
                        tree['relation'] = '=='
                    if not self.names is None:
                        tree['name'] = self.names[j]
                    break
                offset += len(encoders)
        return tree
                    
