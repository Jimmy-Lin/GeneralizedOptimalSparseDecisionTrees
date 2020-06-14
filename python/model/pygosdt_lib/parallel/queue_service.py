from itertools import cycle
from random import shuffle, randint
from multiprocessing import Value
from math import ceil, floor
from time import time, sleep

from model.pygosdt_lib.parallel.channel import Channel, EndPoint
from model.pygosdt_lib.data_structures.heap_queue import HeapQueue
from model.pygosdt_lib.data_structures.task import Task
from model.pygosdt_lib.data_structures.path_cluster import PathCluster
from model.pygosdt_lib.data_structures.capture_cluster import CaptureCluster

def SharedQueueService(queue, degree=1):
    return (__SharedQueueServer__(queue), tuple(__SharedQueueClient__(queue) for _ in range(degree)))

class __SharedQueueServer__:
    def __init__(self, queue):
        self.queue = queue
    
    def serve(self):
        pass

class __SharedQueueClient__:
    def __init__(self, queue):
        self.queue = queue

    def synchronize(self):
        pass

    def push(self, element, block=False):
        self.queue.push(element)

    def pop(self, block=False):
        return self.queue.pop()

    def length(self, local=True):
        return len(self.queue)

    def __repr__(self):
        return repr(self.queue)

    def __str__(self):
        return str(self.queue)

    def flush(self):
        pass

def QueueService(queue=None, degree=1, synchronization_cooldown=0, alpha=0.05, beta=0.05, manager=None):
    if queue == None:
        queue = HeapQueue()

    if queue.lock != None:
        return SharedQueueService(queue, degree=degree)

    if degree > 1:
        global_length = Value('i', len(queue))
        clients = []
        server_endpoints = []
        for i in range(degree):
            client_endpoint, server_endpoint = Channel(duplex=True, channel_type='pipe')

            client = __QueueClient__(queue.new(), client_endpoint, global_length, 
                degree=degree, synchronization_cooldown=synchronization_cooldown, beta=beta)
            clients.append(client)

            server_endpoints.append(server_endpoint)

        server = __QueueServer__(queue, server_endpoints, global_length, alpha=alpha)

    else:
        global_length = Value('i', len(queue))
        clients = []
        server_endpoints = []
        for i in range(degree):
            client_endpoint, server_endpoint = Channel(duplex=True, channel_type='pipe')

            client = __QueueClient__(queue, client_endpoint, global_length, 
                degree=degree, synchronization_cooldown=synchronization_cooldown, beta=beta)
            client.online = False
            clients.append(client)

            server_endpoints.append(server_endpoint)

        server = __QueueServer__(queue.new(), server_endpoints, global_length, alpha=alpha)
        server.online = False

    return (server, tuple(clients))

class __QueueServer__:
    def __init__(self, queue, endpoints, global_length, alpha=1.0):
        self.queue = queue
        self.endpoints = endpoints
        self.global_length = global_length
        self.clusters = tuple(PathCluster() for _ in endpoints)
        self.alpha = alpha
        self.online = True

    def optimal_cluster_index(self, element):
        optimum = -float('Inf')
        optimizer = randint(0, len(self.clusters)-1)

        max_size = -float('Inf')
        min_size = float('Inf')
        for cluster in self.clusters:
            max_size = max(max_size, len(cluster))
            min_size = min(min_size, len(cluster))

        # Maximize proximity to cluster to prioritize cache locality (based on approximate measure of knowledge over dependencies)
        # Minimuze cluster size to prioritize even task distribution
        for i, cluster in enumerate(self.clusters):
            if self.alpha == 0:
                score = - (len(cluster) - min_size) / max((max_size - min_size), 1)
            else:
                score = self.alpha * cluster.proximity(element) - (len(cluster) - min_size) / max((max_size - min_size), 1)
            if score > optimum:
                optimum = score
                optimizer = i
        return optimizer


    def serve(self):
        '''
        Call periodiclly to transfer elements along 3-stage pipeline
        Stage 1: inbound
          messages aggregated from processes in FIFO order
        Stage 2: priority
          messages sorted by priority using heapq module
        Stage 3: outbound
          sorted messages ready for distribution
        '''

        modified = False
        if self.online:
            # shuffle(self.endpoints)
            filtered = 0
            seen = set()
            # Transfer from inbound queue to priority queue
            for i, producer in enumerate(self.endpoints):
                cluster = self.clusters[i]
                while not self.queue.full():
                    element = producer.pop(block=False)
                    if element == None:
                        break

                    cluster.remove(element)
                    key = element if not type(element) == Task else element.key()
                    if not key in seen:
                        seen.add(key)
                        self.queue.push(element)
                    else:
                        filtered += 1
                    
            with self.global_length.get_lock():
                self.global_length.value -= filtered

            # print("Server queue has {} items".format(len(self.queue)))

            modified = len(self.queue) > 0
            while not self.queue.empty():
                element = self.queue.pop()
                optimal_index = self.optimal_cluster_index(element)
                self.clusters[optimal_index].add(element)
                self.endpoints[optimal_index].push(element, block=False)

        return modified

    def flush(self):
        self.serve()

class __QueueClient__:
    def __init__(self, queue, endpoint, global_length, degree=1, synchronization_cooldown=0, beta = 0.01):
        self.queue = queue
        self.endpoint = endpoint
        self.global_length = global_length
        self.synchronization_cooldown = synchronization_cooldown

        self.beta = beta
        self.degree = degree
        self.delta = 0
        self.last_synchronization = 0
        self.online = True
        self.visualizer = None

    def synchronize(self):
        if not self.online:
            return
        if time() > self.last_synchronization + self.synchronization_cooldown:
            self.last_synchronization = time()
        else:
            return

        # Run distribution check
        if self.delta != 0:
            with self.global_length.get_lock():
                self.global_length.value += self.delta
            self.delta = 0

        target = min(max(self.global_length.value / self.degree, 1), self.global_length.value)
        tolerance = floor(self.global_length.value * self.beta)
        lower_target = min(max(floor(target) - tolerance, 1), self.global_length.value)
        upper_target = min(max(ceil(target) + tolerance, 1), self.global_length.value)
        if len(self.queue) > upper_target:
            while len(self.queue) > target and len(self.queue) > 1:
                element = self.queue.pop()
                self.endpoint.push(element, block=False)
                if self.visualizer != None:
                    self.visualizer.send(('queue', 'pop', element))
        while True:
            element = self.endpoint.pop(block=False)
            if element == None:
                break
            self.queue.push(element)
            if self.visualizer != None:
                self.visualizer.send(('queue', 'push', element))

    def push(self, element, block=False):
        '''
        Pushes object into pipeline
        Returns True if successful
        Returns False if unsuccessful
        '''
        
        self.queue.push(element)
        self.delta += 1
        if self.visualizer != None:
            self.visualizer.send(('queue', 'push', element))
        self.synchronize()
        return

    def pop(self, block=False):
        '''
        Pops object from pipeline
        Returns (priority, element) if successful
        Returns (None, None) if unsuccessful
        '''
        self.synchronize()
        element = self.queue.pop()
        if element != None:
            self.delta -= 1
            if self.visualizer != None:
                self.visualizer.send(('queue', 'pop', element))
        return element

    def length(self, local=True):
        if local:
            return len(self.queue)
        else:
            return self.global_length.value

    def __str__(self):
        return str(self.queue)

    def flush(self):
        self.synchronize()
