from math import floor
from multiprocessing import Lock
from multiprocessing.managers import ListProxy

class HeapQueue:
    def __init__(self, queue=None, limit=None):
        if type(queue) == ListProxy:
            self.queue = queue
            self.lock = Lock()
        else:
            self.queue = queue if queue != None else []
            self.lock = None
        self.limit = limit

    def push(self, element):  
        if self.lock == None:
            if self.full():
                return False
            self.queue.append(element)
            self.bubble_up()
            return True
        else:
            with self.lock:
                if self.full():
                    return False
                self.queue.append(element)
                self.bubble_up()
                return True

    def pop(self):
        
        if self.lock == None:
            if self.empty():
                return None
            element = self.queue[0]
            if len(self.queue) > 1:
                self.queue[0] = self.queue.pop()
                self.bubble_down()
            else:
                self.queue.pop()
            return element
        else:
            with self.lock:
                if self.empty():
                    return None
                element = self.queue[0]
                if len(self.queue) > 1:
                    self.queue[0] = self.queue.pop()
                    self.bubble_down()
                else:
                    self.queue.pop()
                return element

    def bubble_up(self):
        i = len(self.queue) -1
        while i > 0 and (self.queue[i] < self.queue[floor((i - 1) / 2)]):
            tmp = self.queue[i]
            self.queue[i] = self.queue[floor((i - 1) / 2)]
            self.queue[floor((i - 1) / 2)] = tmp
            i = floor((i - 1) / 2)

    def bubble_down(self):
        i = 0
        while (2 * i + 1 < len(self.queue) and self.queue[i] > self.queue[2 * i + 1]) or (2 * i + 2 < len(self.queue) and self.queue[i] > self.queue[2 * i + 2]):
            if (2 * i + 2 >= len(self.queue)) or (self.queue[2 * i + 1] <= self.queue[2 * i + 2]):
                j = 2 * i + 1
            else:
                j = 2 * i + 2
            tmp = self.queue[j]
            self.queue[j] = self.queue[i]
            self.queue[i] = tmp
            i = j

    def empty(self):
        return len(self.queue) == 0

    def full(self):
        return self.limit != None and len(self.queue) >= self.limit

    def new(self):
        return HeapQueue(limit=self.limit)

    def __len__(self):
        return len(self.queue)

    def __str__(self):
        return str(self.queue)


