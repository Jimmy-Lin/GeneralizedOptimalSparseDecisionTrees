from time import time, sleep, perf_counter

from .actor import Actor
from .channel import Channel
class Cluster:
    def __init__(self, task, services, size=1, server_period=0.0):
        self.task = task
        self.size = size
        self.server_period = server_period

        # Each service is structured as:
        # tuple( server_interface, tuple( client_1_interface_1, client_2_interface, ... ) )

        # Extract a bundle of server interfaces out of each service
        self.server_bundle = tuple( service[0] for service in services)

        # Extract bundles of client interfaces out of each service
        self.client_bundles = tuple( tuple( service[1][client_id] for service in services ) for client_id in range(self.size))

    def compute(self, max_time):

        def server_task(server_id, services):
            while True:  # Continue servicing as it is not alone
                # sleep(self.server_period)
                for service in services:
                    service.serve()

        (output_consumer, output_producer) = Channel(write_lock=True, channel_type='queue')
        clients = tuple(Actor(i, self.client_bundles[i], self.task, output_channel=output_consumer, actor_type='process') for i in range(0, self.size))
        server = Actor(self.size, self.server_bundle, server_task, output_channel=output_consumer, actor_type='process')

        server.start()
        for client in clients:
            client.start()

        # Permit GC on local service resources now that they have been transferred to their respective subprocesses
        self.server_bundle = None
        self.client_bundles = None

        start_time = perf_counter()
        result = None
        while perf_counter() - start_time < max_time:
            result = output_producer.pop(block=False)
            if result != None:
                break
        duration = perf_counter() - start_time

        # Possibly terminate daemon actors?
        server.actor.terminate()
        for client in clients:
            client.actor.terminate()

        return result, duration
