from multiprocessing import Lock, Pipe, Queue, Value
from collections import deque
from threading import Thread, Event
from queue import Full, Empty

# Wrapper class pair around the multiprocessing Pipe class made to resemble the multiprocessing Queue class
# This implementation exposes separate producer and consumer ends to prevent replication of excess file descriptors

def Channel(read_lock=False, write_lock=False, duplex=False, buffer_limit=None, channel_type=None):
    if channel_type == None:
        channel_type = 'queue' if read_lock or write_lock else 'pipe'
    if channel_type == 'pipe':
        return PipeChannel(read_lock=read_lock, write_lock=write_lock, duplex=duplex, buffer_limit=buffer_limit)
    elif channel_type == 'queue':
        return QueueChannel(read_lock=read_lock, write_lock=write_lock, duplex=duplex, buffer_limit=buffer_limit)
    else:
        raise Exception("ChannelException: Invalid Channel Type {}".format(channel_type))

def PipeChannel(read_lock=False, write_lock=False, duplex=False, buffer_limit=None):
    if duplex:
        (connection_a, connection_b) = Pipe(True)

        consumer_a = __ConnectionConsumer__(connection_a, lock=Lock() if read_lock or write_lock else None, buffer_limit=buffer_limit)
        producer_a = __ConnectionProducer__(connection_a, lock=Lock() if read_lock or write_lock else None)

        consumer_b = __ConnectionConsumer__(connection_b, lock=Lock() if read_lock or write_lock else None, buffer_limit=buffer_limit)
        producer_b = __ConnectionProducer__(connection_b, lock=Lock() if read_lock or write_lock else None)

        endpoint_a = __ProducerConsumer__(consumer_a, producer_a)
        endpoint_b = __ProducerConsumer__(consumer_b, producer_b)
        return (endpoint_a, endpoint_b)
    else:
        (read_connection, write_connection) = Pipe(False)
        consumer = __ConnectionConsumer__(write_connection, lock=Lock() if write_lock else None, buffer_limit=buffer_limit)
        producer = __ConnectionProducer__(read_connection, lock=Lock() if read_lock else None)
        return (consumer, producer)

def QueueChannel(read_lock=False, write_lock=False, duplex=False, buffer_limit=None):
    if duplex:
        if buffer_limit == None:
            forward_queue = Queue()
            backward_queue = Queue()
        else:
            forward_queue = Queue(buffer_limit)
            backward_queue = Queue(buffer_limit)

        consumer_a = __QueueConsumer__(forward_queue)
        producer_a = __QueueProducer__(backward_queue)

        consumer_b = __QueueConsumer__(backward_queue)
        producer_b = __QueueProducer__(forward_queue)

        endpoint_a = __ProducerConsumer__(consumer_a, producer_a)
        endpoint_b = __ProducerConsumer__(consumer_b, producer_b)
        return (endpoint_a, endpoint_b)
    else:
        if buffer_limit == None:
            queue = Queue()
        else:
            queue = Queue(buffer_limit)
        
        consumer = __QueueConsumer__(queue)
        producer = __QueueProducer__(queue)
        return (consumer, producer)

def EndPoint(consumer, producer):
    return __ProducerConsumer__(consumer, producer)

class __ProducerConsumer__:
    def __init__(self, consumer, producer):
        self.consumer = consumer
        self.producer = producer

    def push(self, element, block=False):
        return self.consumer.push(element, block=block)
    
    def pop(self, block=False):
        return self.producer.pop(block=block)

    def full(self):
        return self.consumer.full()

class __ConnectionConsumer__:
    def __init__(self, connection, lock=None, buffer_limit=None):
        self.lock = lock
        self.connection = connection
        self.buffer = deque([])
        self.flushing = Event()
        self.thread = None
        self.buffer_limit = buffer_limit
    
    def push(self, element, block=False):
        if self.buffer_limit != None and len(self.buffer) > self.buffer_limit:
            return False
        
        self.buffer.appendleft(element)
        if not self.flushing.is_set():
            self.thread = Thread(target=self.__flush__)
            self.thread.daemon = True
            self.thread.start()
        
        if block:
            self.thread.join()
        
        return True
        
    def full(self):
        if self.buffer_limit == None:
            return False
        else:
            return len(self.buffer) > self.buffer_limit

    def __flush__(self):
        self.flushing.set()
        while True:
            try:
                element = self.buffer.pop()
                self.__push__(element)
            except IndexError: # Buffer is empty
                break
        self.flushing.clear()
    
    def __push__(self, element):
        if self.lock != None:
            with self.lock:
                try:
                    self.connection.send(element)
                except Exception:
                    pass
        else:
            try:
                self.connection.send(element)
            except Exception:
                pass

class __ConnectionProducer__:
    def __init__(self, connection, lock):
        self.lock = lock
        self.connection = connection
        self.buffer = deque([])

    def pop(self, block=False):
        if len(self.buffer) == 0:
            timeout = None if block else 0
            if self.lock != None:
                with self.lock:
                    while True:
                        try:
                            while self.connection.poll(timeout):
                                self.buffer.appendleft(self.connection.recv())
                        except EOFError:
                            pass
                        finally:
                            if len(self.buffer) > 0 or not block:
                                break
            else:
                while True:
                    try:
                        while self.connection.poll(timeout):
                            self.buffer.appendleft(self.connection.recv())
                    except EOFError:
                        pass
                    finally:
                        if len(self.buffer) > 0 or not block:
                            break
        if len(self.buffer) == 0:
            return None
        else:
            return self.buffer.pop()

class __QueueConsumer__:
    def __init__(self, queue):
        self.queue = queue
    def push(self, element, block=False):
        try:
            self.queue.put(element, block)
            return True
        except Full:
            return False
    def full(self):
        return False

class __QueueProducer__:
    def __init__(self, queue):
        self.queue = queue

    def pop(self, block=False):
        try:
            element = self.queue.get(block)
            return element
        except Empty:
            return None
