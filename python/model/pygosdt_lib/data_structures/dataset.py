import numpy as np
from functools import reduce

from model.pygosdt_lib.data_structures.vector import Vector

# third-party imports
import pandas as pd
from sklearn.utils import shuffle

# Summary: Read in the datasets and returns a Pandas dataframe
# Input:
#   path: relative path to csv path
#   sep: separation character of csv
# Output:
#   dataset: a Pandas dataframe containing the dataset at given path
def read_dataframe(path, sep=None, randomize=False):
    if sep == None:
        with open(path) as f:
            first_line = f.readline()
        if len(first_line.split(';')) > len(first_line.split(',')):
            sep = ';'
        elif len(first_line.split(' ')) > len(first_line.split(',')):
            sep = ' '
        else:
            sep = ','

    dataframe = pd.DataFrame(pd.read_csv(path, sep=sep))
    if randomize:
        dataframe = shuffle(dataframe)
    return dataframe

# There must be a compression threshold beyond which it is no longer worth compressing
class DataSet:
    def read(path, sep=None, randomize=False):
        dataframe = read_dataset(path, sep=sep, randomize=randomize)
        X = dataframe[:, :-1]
        y = dataframe[:, -1]
        return Dataset(X, y)

    def __init__(self, X, y, compression=True, objective='accuracy', accuracy_weight=1.0):
        self.original = (X, y)
        self.split_cache = {}
        self.label_distribution_cache = {}
        self.compression = compression
        self.objective = objective
        self.accuracy_weight = accuracy_weight

        (n, m) = X.shape
        self.sample_size = n  # Number of rows (non-unique)
        self.y = Vector(y)

        # Performs a compression by aggregating groups of rows with identical features
        z, rows, columns = self.__summarize__(X, y)
        self.compression_rate = n / len(rows) # Amount of row reduction achieved
        self.rows = rows # Tuple of unique rows
        self.columns = columns # Tuple of columns (Shortened by the compression ratio)
        self.z = z # Tuple of label distributions per unique row
        self.height = len(rows) # Number of rows (unique)
        self.width = len(columns) # Number of columns

        # Precomputes column ranking, other methods use the ranking to prioritize splits
        self.gini_index = self.__gini_reduction_index__(rows, columns, y)


        if self.compression:
            self.minimum_group_size = min(sum([i]) for i in range(self.height))
            self.maximum_group_size = max(sum(z[i]) for i in range(self.height))

    def split(self, j, capture=None):
        key = (j, capture)
        cache = self.split_cache
        if not key in cache:
            if capture == None:
                capture = Vector.ones(self.height)
            cache[key] = (~self.columns[j] & capture, self.columns[j] & capture)
        return cache[key]

    def splits(self, capture=None):
        if capture == None:
            capture = Vector.ones(self.height)
        return ( (j, *self.split(j, capture=capture)) for j in self.gini_index)

    # Count various frequencies of the y-labels over a capture set
    def label_distribution(self, capture=None):
        key = capture
        cache = self.label_distribution_cache
        if not key in cache:

            if self.compression:
                if capture == None:
                    capture = Vector.ones(self.height)
                (zeros, ones, minority, majority) = reduce(
                    lambda x, y: tuple(sum(z) for z in zip(x, y)),
                    ( (*self.z[i], min(self.z[i]), max(self.z[i])) for i in range(self.height) if capture[i] == 1),
                    (0, 0, 0, 0))
                cache[key] = (zeros + ones, zeros, ones, minority, majority)
            else:
                if capture == None:
                    capture = Vector.ones(self.sample_size)
                zeros = self.weights[0] * (capture & (~self.y)).count() # Rows that are both captured and labeled false
                ones = self.weights[1] * (capture & self.y).count() # Rows that are both captured and labeled True
                if self.objective == 'balanced_accuracy' or self.objective == 'weighted_accuracy':
                    minority = 0
                    majority = 0
                    for i in range(len(self.z)):
                        if capture[i] == 1:
                            if self.z[i] == 1:
                                majority += self.weights[self.y[i]]
                            else:
                                minority += self.weights[self.y[i]]
                else:
                    minority = (capture & (~self.z)).count() # Rows that are both captured and not in majority
                    majority = (capture & self.z).count() # Rows that are both capures and in majority
                total = zeros + ones
                cache[key] = (total, zeros, ones, minority, majority)
        return cache[key]
        

    def __summarize__(self, X, y):
        """
        Summarize the training dataset by recognizing that rows with equivalent features can be
        aggregated into groups. We maintain the label frequencies of each group in 'z'.

        Return values:
        z is a k-tuple of 2-tuples: eg. ((32, 44), (1, 21), (90, 83), ...)
         - 2-tuples contain the frequency of y==0 and y==1 labels respectively within an equivalent group
         - The k-tuple orders sub-tuples in the same order as the rows (which now represent equivalent groups)
        rows is the bit-vector rows filtered down to only unique ones: eg. (<b001010010>, <b001111110>, ...)
        columns is the bit-vector columns shortened to only the unique rows: eg. (<b0010010001010>, <b0010101111110>, ...)
        """
        (n, m) = X.shape
        # Vectorize features and labels for bitvector operations
        columns = tuple(Vector(X[:, j]) for j in range(m))
        rows = tuple(Vector(X[i, :]) for i in range(n))

        zeros_total = float(sum( int(y[i] == 0) for i in range(n)))
        ones_total = float(sum( int(y[i] == 1) for i in range(n)))
        if self.objective == 'accuracy':
            self.weights = [1.0, 1.0]
        elif self.objective == 'balanced_accuracy':
            zeros_weight = 0.5 * (n / zeros_total)
            ones_weight = 0.5 * (n / ones_total)
            self.weights = [zeros_weight, ones_weight]
            # [0.930611694960927, 1.080569461827284]
        elif self.objective == 'weighted_accuracy':
            balance_factor = self.accuracy_weight
            zeros_weight = zeros_total / (balance_factor * ones_total + zeros_total)  * (n / zeros_total)
            ones_weight = balance_factor * ones_total / (balance_factor * ones_total + zeros_total) * (n / ones_total)
            self.weights = [zeros_weight, ones_weight]
        else:
            print(self.objective == 'accuracy')
            raise Exception("Unrecognized objective name '{}'".format(self.objective))

        print('weights = ', self.weights)
        z = {}
        for i in range(n):
            row = rows[i]
            if z.get(row) == None:
                # z stores a tuple for each unique row (equivalent point set)
                # the tuple stores (in order) the frequency of labels 0 and 1
                z[row] = [0, 0]

            # Increment the corresponding label frequency in the equivalent point set
            z[row][y[i]] += self.weights[y[i]]

        reduced_rows = list(z.keys())
        compression_rate = len(rows) / len(reduced_rows)
        print("Row Compression Factor: {}".format(round(compression_rate, 3)))

        if self.compression:
            # Greedy method of ordering rows to maximize trailing zeros in columns
            # This makes column vector have leading zeros, reducing memory comsumption
            reduced_columns = tuple(Vector(row[j] for row in reduced_rows) for j in range(m))
            weights = [ column.count() for column in reduced_columns ]
            reduced_rows.sort(key = lambda row : sum(weights[j] * row[j] for j in range(m)), reverse=True)

            # Convert to tuples for higher cache locality
            z = tuple( tuple(z[row]) for row in reduced_rows )
            columns = tuple(Vector(row[j] for row in reduced_rows) for j in range(m))
            rows = tuple(reduced_rows)
        else:
            majority = [ int(z[rows[i]][y[i]] == max(z[rows[i]])) for i in range(n)]
            z = Vector(majority)

        return z, rows, columns

    def __reduction__(self, captures, column, y):
        """
        computes the weighted sum of Gini coefficients of bisection subsets by feature j
        reference: https://en.wikipedia.org/wiki/Gini_coefficient
        """
        (negative_total, _zeros, ones, _minority, _majority) = self.label_distribution(captures & ~column)
        p_1 = round(ones / negative_total if negative_total > 0 else 0, 10)

        (positive_total, _zeros, ones, _minority, _majority) = self.label_distribution(captures & column)
        p_2 = round(ones / positive_total if positive_total > 0 else 0, 10)

        # Degree of inequality of labels in samples of negative feature
        gini_1 = 2 * p_1 * (1 - p_1)
        # Degree of inequality of labels in samples of positive feature
        gini_2 = 2 * p_2 * (1 - p_2)
        # Base inequality minus inequality
        reduction = negative_total * gini_1 + positive_total * gini_2
        return reduction

    def __gini_reduction_index__(self, rows, columns, y, captures=None):
        """
        calculate the gini reduction by each feature
        return the rank of by descending
        """
        (n, m) = (len(rows), len(columns))
        captures = Vector.ones(n) if captures == None else captures
        # Sets the subset we compute probability over
        (total, _zeros, ones, _minority, _majority) = self.label_distribution(captures)
        p_0 = ones / total if total > 0 else 0

        gini_0 = 2 * p_0 * (1 - p_0)

        # The change in degree of inequality when splitting the capture set by each feature j
        # In general, positive values are similar to information gain while negative values are similar to information loss
        # All values are negated so that the list can then be sorted by descending information gain
        reductions = [-(gini_0 - self.__reduction__(captures, column, y) / total) for column in columns]
        order_index = np.argsort(reductions)

        # print("Negative Gini Reductions: {}".format(tuple(reductions)))
        # print("Negative Gini Reductions Index: {}".format(tuple(order_index)))

        return tuple(order_index)
