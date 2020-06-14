from multiprocessing import Process, Value
from threading import Thread, Event
from os import system, getpid
from subprocess import check_call, DEVNULL, STDOUT
from traceback import print_exc


def Client(id, services, task, client_type='process', output_channel=None):
    return Actor(id, services, task, actor_type=client_type, output_channel=output_channel)

def Server(id, services, server_type='process', output_channel=None):
    return Actor(id, services, server_task, actor_type=server_type, output_channel=output_channel)

def LocalClient(id, services, task, output_channel=None):
    return Actor(id, services, task, actor_type='local', output_channel=output_channel)

def LocalServer(id, services, output_channel=None):
    return Actor(id, services, server_task, actor_type='local', output_channel=output_channel)

def ThreadClient(id, services, task, client_type='thread', output_channel=None):
    return Actor(id, services, task, actor_type=client_type, output_channel=output_channel)

def ThreadServer(id, services, output_channel=None):
    return Actor(id, services, server_task, actor_type='thread', output_channel=output_channel)

def server_task(id, services):
    while True: # Continue servicing as it is not alone
        for service in services:
            service.serve()

def Actor(actor_id, services, task, actor_type='process', output_channel=None):
    if actor_type == 'process':
        return __ProcessActor__(actor_id, services, task, output_channel=output_channel)
    elif actor_type == 'thread':
        return __ThreadActor__(actor_id, services, task, output_channel=output_channel)
    elif actor_type == 'local':
        return __LocalActor__(actor_id, services, task, output_channel=output_channel)
    else:
        raise Exception("ActorException: Invalid Actor Type {}".format(actor_type))

class __LocalActor__:
    def __init__(self, actor_id, services, task, output_channel=None):
        self.__run__ = lambda: task(actor_id, services)
        self.output_channel = output_channel

    def start(self, block=False):
        result = self.__run__()
        if self.output_channel != None:
            self.output_channel.push(result, block=False)

    def join(self):
        return

    def is_alive(self):
        return False

class __ProcessActor__:
    def __init__(self, actor_id, services, task, output_channel=None):
        self.actor = Process(target=self.__run__, args=(actor_id, services, task))
        self.actor.daemon = True
        self.output_channel = output_channel

    def __run__(self, actor_id, services, task):
        # Attempt to pin process to CPU core using taskset if available
        taskset_enabled = (system("command -v taskset") != 256)
        if taskset_enabled:
            try:
                print("taskset", "-cp", str(actor_id), str(getpid()))
                check_call(["taskset", "-cp", str(actor_id), str(getpid())], stdout=DEVNULL, stderr=STDOUT)
            except Exception:
                pass
        try:
            result = task(actor_id, services)
            if self.output_channel != None:
                self.output_channel.push(result, block=False)
        except Exception as e:
            self.exception = e
            print_exc()
            print("ActorException: Actor ID {} caught: {}".format(actor_id, e))

    def start(self, block=False):
        self.actor.start()
        while block and not self.is_alive():
            pass

    def join(self):
        self.actor.join()

    def is_alive(self):
        return self.actor.is_alive()

class __ThreadActor__:
    def __init__(self, actor_id, services, task, output_channel=None):
        self.actor = Thread(target=self.__run__, args=(actor_id, services, task))
        self.actor.daemon = True
        self.output_channel = output_channel
        self.alive = Event()

    def __run__(self, actor_id, services, task):
        self.alive.set()
        try:
            result = task(actor_id, services)
            if self.output_channel != None:
                self.output_channel.push(result, block=False)
        except Exception as e:
            self.exception = e
            print_exc()
            print("ActorException: Actor ID {} caught: {}".format(actor_id, e))
        finally:
            self.alive.clear()

    def start(self, block=False):
        self.actor.start()
        while block and not self.is_alive():
            pass

    def join(self):
        self.actor.join()

    def is_alive(self):
        return self.alive.isSet()
