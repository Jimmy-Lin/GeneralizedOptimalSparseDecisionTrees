import networkx as nx
from bokeh.io import show, output_file
from bokeh.plotting import figure, curdoc
from bokeh.models import Ellipse, MultiLine, HoverTool, BoxZoomTool, ResetTool
from bokeh.models.graphs import from_networkx
from bokeh.server.server import Server
from bokeh.palettes import Spectral4
from threading import Thread, Lock
from time import time, sleep

from model.pygosdt_lib.parallel.channel import Channel
from model.pygosdt_lib.parallel.actor import Actor
from model.pygosdt_lib.data_structures.vector import Vector

class Visualizer:
    def __init__(self, visualizer_id, dataset):
        self.visualizer_id = visualizer_id
        self.path = '/visualizer_{}'.format(self.visualizer_id)
        self.dataset = dataset
        (consumer, producer) = Channel()
        self.consumer = consumer
        self.producer = producer
        self.actor = Actor(visualizer_id, (producer,), self.start, actor_type='thread')
        self.actor.start()

    # Called by the publisher
    def send(self, message):
        self.consumer.push(message)

    def start(self, visualizer_id, services):
        (channel,) = services
        self.channel = channel
        server = Server({self.path: self.visualize}, processes=1, port=5006 + visualizer_id)
        server.start()
        server.io_loop.add_callback(server.show, self.path)
        server.io_loop.start()

    def redraw(self):
        # Construct the graph based on tasks and results
        self.network = nx.DiGraph()

        with self.lock:
            # tiers = {}
            # captures = list(self.tasks)

            # for capture in captures:
            #     if capture in self.results and capture in self.tasks:
            #         result = self.results[capture]
            #         if result.optimizer == None:
            #             self.network.add_node(str(capture),
            #                 optimizer=str(result.optimizer),
            #                 optimum=str(result.optimum),
            #                 color='#669966',
            #                 size= max(0.04 * capture.count() / self.dataset.height, 0.02),
            #                 border=1 if capture in self.tasks else 0
            #             )
            #             for j in self.dataset.gini_index:
            #                 left, right = self.dataset.split(j, capture=capture)
            #                 if left in self.results and right in self.results:
            #                     lowerbound = self.results[left].optimum.lowerbound + self.results[right].optimum.lowerbound
            #                     if lowerbound <= self.results[capture].optimum.upperbound:
            #                         if not left in self.tasks:
            #                             self.network.add_node(str(left),
            #                                 optimizer='???',
            #                                 optimum='???',
            #                                 color='#999999',
            #                                 size= max(0.04 * left.count() / self.dataset.height, 0.02),
            #                                 border=1 if left in self.tasks else 0
            #                             )
            #                             self.network.add_edge(str(capture), str(left),
            #                                 color='#999999',
            #                                 width=1,
            #                             )
            #                             captures.append(left)

            #                         if not right in self.tasks:
            #                             self.network.add_node(str(right),
            #                                 optimizer='???',
            #                                 optimum='???',
            #                                 color='#999999',
            #                                 size= max(0.04 * right.count() / self.dataset.height, 0.02),
            #                                 border=1 if right in self.tasks else 0
            #                             )
            #                             self.network.add_edge(str(capture), str(right),
            #                                 color='#999999',
            #                                 width=1,
            #                             )
            #                             captures.append(right)

            # Full Dependency Graph

            captures = [Vector.ones(self.dataset.height)]

            while len(captures) > 0:
                capture = captures.pop()
                if capture in self.results:
                    result = self.results[capture]
                    self.network.add_node(str(capture),
                        optimizer=str(result.optimizer),
                        optimum=str(result.optimum),
                        color='#339933',
                        size= max(0.05 * capture.count() / self.dataset.height, 0.01),
                        border=1 if capture in self.tasks else 0
                    )
                    if result.optimizer != None:
                        (split, label) = result.optimizer
                        if split != None:
                            left, right = self.dataset.split(split, capture=capture)
                            captures.append(left)
                            captures.append(right)
                            self.network.add_edge(str(capture), str(left),
                                color='#993333',
                                width=1,
                            )
                            self.network.add_edge(str(capture), str(right),
                                color='#333399',
                                width=1,
                            )
                    else:
                        for j in self.dataset.gini_index:
                            left, right = self.dataset.split(j, capture=capture)
                            if left in self.results and right in self.results:
                                lowerbound = self.results[left].optimum.lowerbound + self.results[right].optimum.lowerbound
                                if lowerbound <= self.results[capture].optimum.upperbound:
                                    captures.append(left)
                                    captures.append(right)
                                    self.network.add_edge(str(capture), str(left),
                                        color='#999999',
                                        width=1,
                                    )
                                    self.network.add_edge(str(capture), str(right),
                                        color='#999999',
                                        width=1,
                                    )

        # print("Nodes: {}, Edges: {}".format(self.network.number_of_nodes(), self.network.number_of_edges()))
        
        # bipartite_layout
        # circular_layout
        # kamada_kawai_layout
        # planar_layout
        # random_layout
        # rescale_layout
        # shell_layout
        # spring_layout
        # spectral_layout

        if self.network.number_of_nodes() > 0:
            self.renderer = from_networkx(self.network, nx.circular_layout, scale=1.0, center=(0, 0))
            self.renderer.node_renderer.glyph = Ellipse(
                width='size',
                height='size',
                fill_color='color',
                line_alpha='border'
            )
            self.renderer.edge_renderer.glyph = MultiLine(
                line_color="color",
                line_width='width',
                line_dash='solid',
                line_join='miter'
            )
            self.plot.renderers = [self.renderer]
            # output_file('visualizer_{}.html'.format(self.visualizer_id))
            # show(self.plot)

    def update(self, message):
        (target, command, argument) = message
        if target == 'queue':
            if command == 'push':
                (priority, capture, path) = argument
                with self.lock:
                    self.tasks.add(capture)
            elif command == 'pop':
                (priority, capture, path) = argument
                with self.lock:
                    if capture in self.tasks:
                        self.tasks.remove(capture)
            else:
                raise Exception('VisualizerError: Invalid message {}'.format(message))
        elif target == 'dictionary':
            if command == 'put':
                (capture, result) = argument
                with self.lock:
                    self.results[capture] = result
            else:
                raise Exception('VisualizerError: Invalid message {}'.format(message))
        else:
            raise Exception('VisualizerError: Invalid message {}'.format(message))
        
    def tick(self):
        while True:
            updated = False
            while True:
                message = self.producer.pop(block=False)
                if message != None:
                    updated = True
                    self.update(message)
                else:
                    break
            if updated and time() > self.last_redraw + self.redraw_period:
                self.last_redraw = time()
                self.doc.add_next_tick_callback(self.redraw)

    def visualize(self, doc):
        self.plot = figure(title="OSDT Dependency Graph",
                           x_range=(-1.1, 1.1), y_range=(-1.1, 1.1),
                           sizing_mode='scale_height', tools="", toolbar_location=None, output_backend="webgl")
        self.results = {}
        self.tasks = set()
        self.lock = Lock()
        self.doc = doc
        # self.network = nx.karate_club_graph()
        self.network = nx.DiGraph()
        self.renderer = from_networkx(self.network, nx.circular_layout, scale=1, center=(0, 0))
        self.plot.renderers = [self.renderer]
        node_hover_tool = HoverTool(tooltips=[("index", "@index"), ("optimizer", "@optimizer"), ("optimum", "@optimum")])
        self.plot.add_tools(node_hover_tool)
        self.doc.add_root(self.plot)
        self.redraw_period = 0.1
        self.last_redraw = 0

        thread = Thread(target=self.tick)
        thread.daemon = True
        thread.start()
