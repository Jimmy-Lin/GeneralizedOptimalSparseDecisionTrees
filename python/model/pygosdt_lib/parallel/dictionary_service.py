from time import time, sleep
from random import shuffle
from copy import deepcopy

from model.pygosdt_lib.data_structures.hash_table import HashTable
from model.pygosdt_lib.data_structures.vector import Vector
from model.pygosdt_lib.data_structures.result import Result
from model.pygosdt_lib.data_structures.interval import Interval
from model.pygosdt_lib.data_structures.prefix_tree import PrefixTree
from model.pygosdt_lib.parallel.channel import Channel

def SharedDictionaryService(table, propagator=None, degree=1):
    return (__SharedDictionaryServer__(table), tuple(__SharedDictionaryClient__(table) for _ in range(degree)))

class __SharedDictionaryServer__:
    def __init__(self, table):
        self.table = table

    def serve(self):
        pass

class __SharedDictionaryClient__:
    def __init__(self, table):
        self.table = table

    def synchronize(self):
        pass

    def has(self, key):
        return key in self.table

    def get(self, key, block=False):
        return self.table.get(key)

    def put(self, key, value, block=False):
        # print(type(self.table))
        # with self.table.table.get_lock():
        #     if self.table.accepts(key, value):
                
        self.table[key] = value

    def shortest_prefix(self, key):
        if type(self.table) != PrefixTree:
            raise Exception("DictionaryServiceError: DictionaryTable of internal type {} does no support prefix queries".format(type(self.table)))
        return self.table.shortest_prefix(key)

    def longest_prefix(self, key):
        if type(self.table) != PrefixTree:
            raise Exception("DictionaryTableError: DictionaryTable of internal type {} does no support prefix queries".format(type(self.table)))
        return self.table.longest_prefix(key)

    def __getitem__(self, key):
        return self.table.get(key, block=False)

    def __setitem__(self, key, value):
        self.table.put(key, value, block=False)

    def __contains__(self, key):
        return self.has(key)

    def __str__(self):
        return str(self.table)
    
    def __repr__(self):
        return repr(self.table)

    def __len__(self):
        return len(self.table)

    def flush(self):
        pass


def DictionaryService(table=None, propagator=None, synchronization_cooldown=0, degree=1):
    if table == None:
        table = HashTable()

    if table != None and table.lock != None:
        return SharedDictionaryService(table, propagator=propagator, degree=degree)
    
    clients = []
    server_endpoints = []
    for i in range(degree):
        client_endpoint, server_endpoint = Channel(duplex=True, channel_type='pipe')
        client = __DictionaryClient__(deepcopy(table), client_endpoint, propagator=propagator, synchronization_cooldown=synchronization_cooldown)
        clients.append(client)
        server_endpoints.append(server_endpoint)

    server = __DictionaryServer__(table, server_endpoints)

    if degree <= 1:
        server.online = False
        for client in clients:
            client.online = False

    return (server, tuple(clients))


class __DictionaryServer__:
    def __init__(self, table, endpoints):
        self.table = table
        self.updates = {}
        self.endpoints = endpoints
        self.online = True
        self.dataset = None


    def solved(self, capture, path):
        result = self.table.get(capture)
        if result == None:
            return False
        elif result.optimizer == None:
            return False
        elif result.optimizer[1] != None:
            return True
        elif result.optimizer[0] != None:
            j = result.optimizer[0]
            left_capture, right_capture = self.dataset.split(j, capture)
            return self.solved(left_capture, path + (j, 'L')) and self.solved(right_capture, path + (j, 'R'))

    # Service routine called by server
    def serve(self):
        '''
        Call periodically to transfer elements along 3-stage pipeline
        Stage 1: inbound
            entries aggregated from processes in FIFO order
            repeated entries are discarded, new entries advance to stage 2
        Stage 2: buffers
            new entries stored in buffers (1 replica per subscriber)
            buffered elements transfer to outbound queue when possible
        Stage 3: outbound
            outbound entries ready for consumption
        '''
        modified = False
        if self.online:
            shuffle(self.endpoints)

            self.updates.clear()
            # Transfer from inbound queue to broadcast buffers (if the entry is new)
            for endpoint in self.endpoints:
                while True:
                    element = endpoint.pop(block=False)
                    if element == None:
                        break
                    (key, value) = element
                    if self.table.accepts(key, value):
                        # print("DictionaryTable Update table[{}] from {} to {}".format(str(key), str(previous_value), str(value)))
                        self.updates[key] = value
                        self.table[key] = value
                    else:
                        # print("Rejected DictionaryTable Update table[{}] = from {} to {}".format(str(key), str(self.table[key]), str(value)))
                        pass

            # if self.dataset != None:
            #     print("Server Solved", self.solved(Vector.ones(self.dataset.height), tuple()))

            for key, value in self.updates.items():
                for endpoint in self.endpoints:
                    endpoint.push((key, value), block=False)

        return modified

    def flush(self):
        self.serve()

class __DictionaryClient__:
    def __init__(self, table, endpoint, propagator=None, synchronization_cooldown=0):
        self.table = table
        self.endpoint = endpoint
        self.synchronization_cooldown = synchronization_cooldown
        self.propagator = propagator
        self.last_synchronization = 0
        self.online = True
        self.visualizer = None

    def synchronize(self):
        '''
        Receives broadcasted entries from pipeline into local cache
        '''

        if not self.online:
            return

        if time() > self.last_synchronization + self.synchronization_cooldown:
            self.last_synchronization = time()
        else:
            return
        
        while True:
            element = self.endpoint.pop(block=False)
            if element == None:
                break
            (key, value) = element
            if self.table.accepts(key, value):
                # Perform extrapolation
                if self.propagator != None and type(value) == Result and not self.propagator.tracking(key):
                    new_value = self.propagator.converge(key, value, self.table)
                    if new_value.optimum == value.optimum:
                        pass
                        # self.propagator.track(key)
                    else:
                        value = new_value

                if self.visualizer != None:
                    self.visualizer.send(('dictionary', 'put', (key, value)))

                self.table[key] = value
                
            # else:
            #     print("Rejected DictionaryTable Update table[{}] = from {} to {}".format(str(key), str(self.table[key]), str(value)))

            


    def has(self, key):
        '''
        Queries local cache for value
        Triggers a refresh upon miss
        Returns True upon hit (after refresh)
        Returns False upon miss (after refresh)
        '''
        self.synchronize()
        return key in self.table

    def get(self, key, block=False):
        '''
        Queries local cache for value
        Triggers a refresh upon miss
        Returns value upon hit (after refresh)
        Returns None upon miss (after refresh)
        '''
        self.synchronize()
        if block:
            while not key in self.table:
                self.synchronize()
        return self.table.get(key)

    def put(self, key, value, block=False):
        '''
        Stores key-value into local cache and sends entry into pipeline
        Returns True if successfully sent into pipeline
        Returns False if unsuccessful in sending to pipeline
        key-value is always written to local cache
        '''

        if self.table.accepts(key, value):
            # Perform extrapolation
            if self.propagator != None and type(value) == Result and not self.propagator.tracking(key):
                new_value = self.propagator.converge(key, value, self.table)
                if new_value.optimum == value.optimum:
                    pass
                    # self.propagator.track(key)
                else:
                    value = new_value

            if self.visualizer != None:
                self.visualizer.send(('dictionary', 'put', (key, value)))

            self.table[key] = value

            return self.endpoint.push((key, value), block=block)

    def converge(self, key, value):
        if self.propagator != None and type(value) == Result and not self.propagator.tracking(key):
            return self.propagator.converge(key, value, self.table)
        else:
            return value
    
    def shortest_prefix(self, key):
        if type(self.table) != PrefixTree:
            raise Exception("DictionaryServiceError: DictionaryTable of internal type {} does no support prefix queries".format(type(self.table)))
        self.synchronize()
        return self.table.shortest_prefix(key)

    def longest_prefix(self, key):
        if type(self.table) != PrefixTree:
            raise Exception("DictionaryTableError: DictionaryTable of internal type {} does no support prefix queries".format(type(self.table)))
        self.synchronize()
        return self.table.longest_prefix(key)

    def __getitem__(self, key):
        return self.get(key, block=False)

    def __setitem__(self, key, value):
        return self.put(key, value, block=False)

    def __contains__(self, key):
        return self.has(key)

    def __str__(self):
        return str(self.table)

    def __len__(self):
        return len(self.table)

    def flush(self):
        self.synchronize()
