from queue import Empty as QueueEmpty, Full as QueueFull

from lib.parallel.cluster import Cluster
from lib.parallel.queue_service import QueueService
from lib.parallel.dictionary_service import DictionaryService
from lib.data_structures.heap_queue import HeapQueue

class Fibonacci:
    def __init__(self, n):
        # Set all global variables, these are statically available to all workers
        self.n = n

    # Task method that gets run by all worker nodes (clients)
    def task(self, worker_id, services):
        (table, queue) = services
        self.table = table
        self.queue = queue
        while not self.complete():
            message = self.queue.pop(block=False)
            if message == None:
                continue
            (priority, n) = message
            # Check status
            result = table[n]


            if result == None:  # New problem
                if n <= 2:  # Base Case
                    output = 1
                    table[n] = output # Memoize resolved problem
                    # print("Fib({}) = {}".format(n, output))
                else: # Recursive Case
                    # print("Fib({}) = ?".format(n))
                    dependencies = (n-1, n-2)
                    table[n] = (n-1, n-2)  # Memoize pending problem
                    if not n-1 in table:
                        queue.push((priority-1, n-1), block=False) # Enqueue subproblem

                    if not n-2 in table:
                        queue.push((priority-2, n-2), block=False)

                    # re-enqueue problem
                    queue.push((priority+0.5, n), block=False)

            else:  # Revisited problem
                if type(result) == int: # Problem solved (No work needed)
                    pass
                elif all(type(table[dependency]) == int for dependency in result): # Dependencies resolved, resolve self
                    output = sum(table[dependency] for dependency in result) # Compute output from subproblems' outputs
                    table[n] = output # Re-memoize as resolved problem
                    # print("Fib({}) = {}".format( n, output))
                else: # Dependencies not resolved, re-enqueue problem
                    queue.push((priority+0.5, n))  # re-enqueue problem
       
        return self.output()

    # Method run by worker nodes to decide when to terminate
    def complete(self):
        return self.table[self.n] != None and type(self.table[self.n]) == int

    # Method for extracting the output 
    def output(self):
        return self.table[self.n]

    def solve(self, workers=1):
        self.workers = workers
        # Shared Data structures that get serviced by servers
        table = DictionaryService(degree=workers)
        queue = QueueService(queue=HeapQueue([(self.n, self.n)]), degree=workers)
        services = (table, queue)

        # Initialize and run the multi-node client-server cluster
        solution = Cluster(self.task, services, size=workers).compute(float('Inf'))
        return solution
