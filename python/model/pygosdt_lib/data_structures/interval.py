import numpy
# Defines the behaviour of a floating point interval
# Used to represent the possible values of the optimal objective to a particular problem

class Interval:
    def __init__(self, lowerbound=None, upperbound=None):
        if lowerbound == None and upperbound == None:
            self.lowerbound = -float('Inf')
            self.upperbound = float('Inf')
        elif lowerbound != None and upperbound == None:
            self.lowerbound = lowerbound
            self.upperbound = lowerbound
        else:
            self.lowerbound = lowerbound
            self.upperbound = upperbound
        if self.lowerbound - numpy.finfo(numpy.float32).eps > self.upperbound:
            raise Exception("Invalid Interval Bounds [{}, {}]".format(lowerbound, upperbound))
        if self.lowerbound == self.upperbound: # Special case to deal with infinities
            self.uncertainty = 0
        else:
            self.uncertainty = self.upperbound - self.lowerbound
    
    def union(self, interval):
        return Interval(min(self.lowerbound, interval.lowerbound), max(self.upperbound, interval.upperbound))

    def intersection(self, interval):
        return Interval(max(self.lowerbound, interval.lowerbound), min(self.upperbound, interval.upperbound))

    def subset(self, interval):
        return self.lowerbound >= interval.lowerbound and self.upperbound <= interval.upperbound and self.uncertainty < interval.uncertainty

    def superset(self, interval):
        return self.lowerbound <= interval.lowerbound and self.upperbound >= interval.upperbound and self.uncertainty > interval.uncertainty
    
    def overlap(self, interval):
        return interval.lowerbound <= self.upperbound and interval.upperbound >= self.lowerbound

    def value(self):
        if self.uncertainty == 0:
            return self.lowerbound
        else:
            return (self.lowerbound, self.upperbound)

    def __or__(self, interval):
        return self.union(interval)

    def __and__(self, interval):
        return self.intersection(interval)

    def __add__(self, interval):
        return self.union(interval)

    def __lt__(self, interval):
        return self.upperbound < interval.lowerbound

    def __le__(self, interval):
        return self.upperbound <= interval.upperbound

    def __gt__(self, interval):
        return self.lowerbound > interval.upperbound

    def __ge__(self, interval):
        return self.lowerbound >= interval.lowerbound

    def __eq__(self, interval):
        if interval == None:
            return False
        return self.lowerbound == interval.lowerbound and self.upperbound == interval.upperbound

    def __ne__(self, interval):
        return not self == interval

    def __str__(self):
        return str(self.value())

    def __len__(self):
        return self.uncertainty
