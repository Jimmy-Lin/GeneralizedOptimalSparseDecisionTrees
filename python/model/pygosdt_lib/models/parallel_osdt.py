from time import time, sleep
from random import random
from memory_profiler import profile
from pickle import dump, load
from multiprocessing import Manager
from math import factorial, ceil, floor
from scipy.special import comb
from sklearn.tree import DecisionTreeClassifier

from model.pygosdt_lib.parallel.cluster import Cluster
from model.pygosdt_lib.parallel.queue_service import QueueService
from model.pygosdt_lib.parallel.dictionary_service import DictionaryService
from model.pygosdt_lib.models.similarity_propagator import SimilarityPropagator
from model.pygosdt_lib.data_structures.prefix_tree import PrefixTree
from model.pygosdt_lib.data_structures.heap_queue import HeapQueue
from model.pygosdt_lib.data_structures.result_table import ResultTable
from model.pygosdt_lib.data_structures.similarity_index import SimilarityIndex, FullIndex
from model.pygosdt_lib.data_structures.result import Result
from model.pygosdt_lib.data_structures.task import Task
from model.pygosdt_lib.data_structures.interval import Interval
from model.pygosdt_lib.data_structures.dataset import DataSet
from model.pygosdt_lib.data_structures.vector import Vector
from model.pygosdt_lib.data_structures.tree import Tree
from model.pygosdt_lib.experiments.logger import Logger
from model.pygosdt_lib.experiments.visualizer import Visualizer

from model.pygosdt_lib.experiments.accuracy import compute_width

# Theorems Applied

# Notes:
# - In this recursive problem formulation, an optimal subtree can be either a leaf or a tree
# - The nodes outside of a subtree doesn't affect the risk contribution of the subtree by any sort of interaction
# - Therefore, a lot of bounds we make on leafs actually generalize to subtrees in this scenario
#   since the nodes outside of a subtree treat the subtree as a black box implementation of a leaf

# Theorem 3 Lower Bound on Leaf Support (Generalized to Subtree Support)
#  - This lowerbound generalizes to the children of all internal nodes, not just leaves
#    If a subtree has a support set of size of k < 2 * lambda,
#    if follows that no pair of leaves of this subtree has support adding up to k >= 2 * lambda
#  - During the downward recursive traversal avoid subproblem pairs if their total support < 2 * lambda

#  The core intuition is that each split increases regularization cost by 1 * lambda
#  To justify the cost, the two subtrees must have a joint miclassification 1 * lambda lower than the parent
#  A subtree is lowerbound by their EQ Minority
#  We know the minority before the split which acts as a baseline
#  
#  
#  - Further, we can create an upperbound on leaf support k <= (parent support) - lambda

# Corollary 6 Leaf Permutation Bound
#  - This generalizes to all subtree combination
#  - In this framework, leaf-sets don't exist but subtrees' subcapture set pairs do
#  - When recursing, we remove all duplicate subcapture sets to avoid unnecessary access to the memoization table
#  - If duplicate subcaptures are generated from different parent (i.e. they don't share the same recursive call)
#    then the memoization table will remove this duplication

# Proposition 8 Equivalent Points Bound
#  - The algorithm uses equivalent points to summarize capture vectors leading to smaller memory usage
#  - Equivalent Points place a lower bound on misclassification for any capture set
#    this lowerbound allows us to potentially prune subproblems without fully computing
#    the true minimal classification with regularization in mind

# TODO:
# Theorem 1 Hierarchical Objective Lowerbound
#  - During the downward recursive traversal,
#    lowerbound the subtree by the inaccuracy of an optimal leaf over the same capture set since it qualifies
#    as a child of such a leaf

# Theorem 2 One Step Lookahead Bound
#  This one is trickier in the new execution framework
#  This would be equivalent to the following:
#   - During the recursive traversal we consider the optimal leaf and all possible splits over the same capture set
#   - Each one is a candidate for the optimal subtree over the capture set C
#   - If we find any one of them has a lowerbound (plus lambda if not a leaf) greater than the current best candidate
#     then that candidate can be ignored.
#   - This means that this problem no longer depends on evaluation of those subproblems but doesn't guarantee that
#     those subproblems are not dependencies for other problems (which means we can't prune them)
#   The best we can do is one of the following
#    - only evaluate one split at a time for stricter comparison, i.e. don't enqueue all subproblems making this more
#      of a DFS like the previous implementation
#    - evaluate all lowerbounds against the optimal leaf (the leaf evaluation is constant) and don't enqueue subproblems
#      that perform worse. This coincidentally is the same as applying Theorem 4 during downward traversal.

# Theorem 4 Lower bound on Incremental Leaf Accuracy (Generalized to Incremental Subtree Accuracy)
#  - This lowerbound generalizes to the children of all internal nodes, not just leaves
#    if we consider an exchange from a leaf to a subtree containing 2 leaves, the risk trade-off is as follows:
#     - The misclassification term decreases by the incremental accuracy from exchanging the leaf to the subtree of 2 leaves (3.10)
#     - The regularization term increases by 1 * lambda
#    However, we can generalize the risk trade-off for exchanging a leaf with a subtree containing k leaves where k >= 2:
#     - The misclassification term decreases by the incremental accuracy from exchanging the leaf to the subtree of k leaves
#     - The regularization term increases by (k-1) * lambda
#     - Note k >= 2, since k = 1 is just an optimal leaf and there's no point exchanging the optimal leaf with itself
#  - For consistency, consider an equivalent upperbound on incemental inaccuracy:
#    (incremental inaccuracy) = -(incremental accuracy)
#    Constraint: incremental inaccuracy <= - (k-1) * lambda
#  - During the downward recursive traversal avoid subproblem pairs if their total lowerbound - (inaccuracy before split) > - lambda
#    lowerbound incremental inaccuracy = (lowerbound after split) - (inaccuracy before split)
#    lowerbound incremental inaccuracy <= incremental inaccuracy <= -lambda
#    (we don't know the number of leaves the optimal subtree will yield so just assume k = 2)
#  - During the upward recursive traversal, ignore subresults pairs if their total inaccuracy > -(k-1) * lambda
#    (we know the optimal subtree in the upward call, so a specific k can be applied)

# Theorem 5 Lowerbound on Leaf Accuracy (Generalized to Lowerbound on Subtree Accuracy)
#  - This lowerbound generalizes to the children of all internal nodes, not just leaves
#    Since optimal trees have a set of k leaves satisfying the leaf accuracy lowerbound of lambda
#    we can assume an optimal subtree containing k leaves has an accuracy lowerbound of k * lambda
#  - For consistency, consider an equivalent upperbound on inaccuracy:
    #  subtree accuracy = correct / n
    #  (subtree inaccuracy) = support - (subtree accuracy)
    #  accuracy = support - inaccuracy
    #  accuracy > k * lambda
    #  support - inaccuracy > k * lambda
    #  support - k * lambda > inaccuracy
    #  inaccuracy < support - k * lambda
    #  inaccuracy + k * lambda < support
    #  In other words, the risk of a subtree should be less than the support
#    Constraint: subtree inaccuracy + k * lambda <= (1 - k * lambda)
#  - During the downward recursive traversal, avoid subproblems if their lowerbound + lambda > support
#    lowerbound + lambda < inaccuracy + lambda <= support
#  - During the upward recursive traversal, ignore subresults inaccuracy + k * lambda > support

# Theorem 7 Similar Support Bound
#  - During any update to the truth table, if problem B exist within neighbourhood of A with within distance w = |CA xor CB|
#     - Their upperbounds may not differ more than w + #(EQ groups) in CA xor CB
#     - Their lowerbounds may not differ more than w
#    More Strictly
#     - Their upperbounds may not differ more than w
#     - Their lowerbounds may not differ more than w

#  - For any two optimal trees A, B with respective capture sets CA, CB and optimal risks RA, RB 
#    |RA - RB| <= hamming_distance(CA, CB) = |CA xor CB|
#  - During the upward recursive traversal, we are informed with an optimal subtree for CA
#    For all subproblems CB satisfying hamming_distance(CA, CB) <= D
#    Apply |RA - RB| <= D to possibly increase the lowerbound of CB by claiming RA - D <= RB
#    If the bound is ineffective the attempt a stricter bound RA - |CA-CB| <= RA - D <= RB
# Changelog
# - stricter base case


class ParallelOSDT:
    interrupt = False
    def __init__(self,
        X, y, regularization,
        configuration=None,
        max_depth=float('Inf'), max_time=float('Inf'),
        verbose=False, log=False, profile=False):

        # Set all global variables, these are statically available to all workers

        # These define the how the algorithm solves the problem
        default_configuration = self.__default_configuration__()
        default_configuration.update(configuration if configuration != None else {})
        self.configuration = default_configuration

        # These define the problem
        self.dataset = DataSet(X, y,
            compression=self.configuration['equivalent_point_compression'],
            objective=self.configuration['objective'],
            accuracy_weight=self.configuration['accuracy_weight'])
        self.lamb = regularization

        # These are additional specifications that the user may pose
        self.max_depth = min(max_depth, self.dataset.width)
        self.max_time = max_time
        self.verbose = verbose
        self.profile = profile
        self.log = log
        self.logger = None
        self.snapshot_cooldown = 0.5
        self.last_snapshot = 0
        self.convergence_logger = None
        self.distribution_logger = None
        self.memory_logger = None
        self.visualizer = None
        self.interrupt = False

        # Global Upperbound
        _total, zeros, ones, minority, _majority = self.dataset.label_distribution()
        self.global_upperbound = min(zeros, ones) / self.dataset.sample_size + self.lamb
        self.global_lowerbound = minority / self.dataset.sample_size + self.lamb


    # @profile
    def snapshot(self):
        if time() > self.last_snapshot + self.snapshot_cooldown:
            self.last_snapshot = time()
        else:
            return
        
        if self.convergence_logger == None:
            self.convergence_logger = Logger(path='data/convergence/worker_{}.csv'.format(self.worker_id), header=['time', 'lowerbound', 'upperbound'])
        root = self.results.get(self.root, block=False)
        if root != None and root.optimum != None:
            self.convergence_logger.log([self.elapsed_time(), root.optimum.lowerbound, root.optimum.upperbound])

        if self.distribution_logger == None:
            self.distribution_logger = Logger(path='data/distribution/worker_{}.csv'.format(self.worker_id), header=['time', 'local_queue_length', 'global_queue_length'])
        self.distribution_logger.log([self.elapsed_time(), self.tasks.length(), self.tasks.length(local=False) ])

        if self.memory_logger == None:
            self.memory_logger = Logger(path='data/memory/worker_{}.csv'.format(self.worker_id), header=['time', 'local_table_size'])
        self.memory_logger.log([self.elapsed_time(), len(self.results)])

    # Task method that gets run by all worker nodes
    def task(self, worker_id, services):
        self.worker_id = worker_id
        start_time = time()
        (tasks, results, prefixes) = services
        self.tasks = tasks
        self.results = results
        self.prefixes = prefixes
        self.logger = Logger(path='logs/worker_{}.log'.format(self.worker_id)) if self.log else None

        # # Switch to Sequential
        # self.tasks.online = False
        # self.results.online = False
        # self.prefixes.online = False

        self.profile_x = []
        self.profile_y = []

        if self.verbose or self.log:
            self.print('Worker {} Starting'.format(self.worker_id))

        # Set up visualizer channel when ready
        if self.visualize_training:
            self.visualizer = Visualizer(self.worker_id, self.dataset)
            self.tasks.visualizer = self.visualizer
            self.results.visualizer = self.visualizer

        while not self.complete() and not self.timeout(): #  and peers.value == self.workers:
            # print("Client {} queue has {} items".format(self.worker_id, len(self.tasks.queue)))

            root = self.results.get(self.root, block=False)
            if not root is None:
                self.profile_x.append(self.elapsed_time())
                self.profile_y.append(root.optimum.upperbound)

            if self.profile: # Data for worker analysis
                self.snapshot()

            # self.results.synchronize()
            # self.prefixes.synchronize()
            task = self.dequeue() # Change to non-blocking since we don't have an alternatve idle task anywyas
            if task == None:
                continue
            
            # Initialize the results table entry if not already initialized, and returns the persisted entry
            # If an entry already exists, simply load the existing one
            result = self.find_or_create_result(task.capture, task.path)

            # An entry with non-zero uncertainty indicates that the precise optimum is not yet found due to lack of information
            # Information is gained everytime a subproblem (direct child or any descendent) gains information
            # The original source of information comes from base cases where the support set can no longer be split so the optimum
            # is calculated directly without searching. From there, information propagates up all ancestry paths in a hierarchical pattern

            if result.optimum.uncertainty > 0: # Recursive case
                # This method creates a generator of all subproblems that need to be solved
                # It also has many side-effects such as:
                #  - Further propagating information through this problem when possible
                #  - Pruning irrelevant subproblems when possible
                #  - Pruning the current problem when possible

                for j in self.dependencies(task, result):
                    left_capture, right_capture = self.dataset.split(j, capture=task.capture)

                    left_path = task.path + (j, 'L')
                    left_priority = self.prioritize(left_capture, left_path)
                    self.enqueue(Task(left_priority, left_capture, left_path))

                    right_path = task.path + (j, 'R')
                    right_priority = self.prioritize(right_capture, right_path)
                    self.enqueue(Task(right_priority, right_capture, right_path))
            else:
                if self.verbose or self.log:
                    self.print('Case: Cached, Problem: {}:{} => {}'.format(task.path, task.capture, result))

        if self.profile:  # Data for worker analysis
            self.last_snapshot = 0
            self.snapshot()

        # print('Worker {} has seen {} problems and {} trees'.format(self.worker_id, len(self.results), self.results[self.root].count))    
        # print('Worker {} has seen {} problems'.format(self.worker_id, len(self.results)))            


        if "profile_output" in self.configuration:
            po = open(self.configuration["profile_output"],"w")
            po.write("time,upperbound\n")
            for x, y in zip(self.profile_x, self.profile_y):
                po.write("{},{}\n".format(x,y))
            po.close()


        self.print('Worker {} Finishing (Complete: {}, Timeout: {})'.format(self.worker_id, self.complete(), self.timeout()))

        return self.output() if self.complete() else None

    def prioritize(self, capture, path):
        priority_metric = self.configuration['priority_metric']
        if priority_metric == 'uniform':
            priority = 0
        elif priority_metric == 'random':
            priority = random()
        elif priority_metric == 'time':
            priority = -time()
        elif priority_metric == 'uncertainty':
            result = self.get(capture, path)
            priority = result.optimum.uncertainty
        elif priority_metric == 'lowerbound':
            result = self.get(capture, path)
            priority = result.optimum.lowerbound
        elif priority_metric == 'upperbound':
            result = self.get(capture, path)
            priority = result.optimum.upperbound
        elif priority_metric == 'support':
            total, _zeros, _ones, _minority, _majority = self.dataset.label_distribution(capture)
            priority = total / self.dataset.sample_size
        elif priority_metric == 'depth':
            priority = -len(path)
        return priority

    def find_or_create_result(self, capture, path):
        if self.configuration['capture_equivalence']:
            key = capture
        else:
            key = (capture, path)
        if not key in self.results:
            bounding_interval, support, majority_label = self.compute_bounds(capture)
            if (self.configuration['accuracy_lowerbound'] and bounding_interval.lowerbound > support):
                optimizer, optimum = None, Interval(float('Inf'))
                result = Result(optimizer=optimizer, optimum=optimum)
            elif (len(path) / 2 >= self.max_depth or capture.count() <= 1 or bounding_interval.uncertainty <= 0 or
                (self.configuration['support_lowerbound'] and 0.5 * support <= self.lamb) or # Insufficient support for splitting
                (self.configuration['incremental_accuracy_lowerbound'] and bounding_interval.uncertainty <= self.lamb)): # Insufficient potential accuracy gain for splitting
                # Base Case: Set up an entry with the optimizer as an optimal leaf and a precise objective
                optimizer, optimum = (None, majority_label), Interval(bounding_interval.upperbound)
                result = Result(optimizer=optimizer, optimum=optimum)
                if self.verbose or self.log:
                    self.print('Case: Base, Problem: {}:{} => {}'.format(path, capture, result))
            else:
                # Recursive Case: Set up an entry with an unknown optimizer and a bounding interval on the objective
                optimizer, optimum = None, bounding_interval
                result = Result(optimizer=optimizer, optimum=optimum)
                if self.verbose or self.log:
                    self.print('Case: Recursive, Problem: {}:{} => {}'.format(path, capture, result))
            self.update(capture, path, result)
        else:
            result = self.get(capture, path)
        return result

    def dependencies(self, task, current_result):
        priority = task.priority
        capture = task.capture
        path = task.path
        # Attempts to solve the problem of which of j in 1:m features to split on or to just use a leaf
        minimum_bounding_interval, minimizing_split, relevant_splits, irrelevant_splits = self.minimize_choice_of_split(capture, path)
        # minimum_bounding_interval = The current best known bound over the objective when considering all the choices
        # minimizing_split = The feature index (None if Leaf is best), that caused the minimum_bounding_interval's upperbound
        #   This is only important if minimum_bounding_interval has 0 uncertainty meaning the optimal choice can be decided
        # relevant_splits = generator of feature indices representing the set of feature indices whose resulting optimal objective lie within
        #   minimum_bounding_interval, and therefore are still worth considering (excludes previously split features)
        # irrelevant_splits = generator of feature indixes represengint the set of indices that do not overlap with minimum_bounding_interval,
        #   and therefore cannot possibly contain the optimal objective

        count = sum( (1 if not d in self.results else self.results[d].count) for d in set(sum([ self.dataset.split(j, capture=task.capture) for j in (tuple(relevant_splits) + tuple(irrelevant_splits)) ], ())) )
        # count = sum( (1 if not left in self.results else self.results[left].count) * (1 if not right in self.results else self.results[right].count) for left, right in ( self.dataset.split(j, capture=task.capture) for j in (tuple(relevant_splits) + tuple(irrelevant_splits)) ) )


        # As this implementation asynchronously tries to solve for the optimal choice, early attempts may lack enough precise bounds on subproblems
        # in order to make a decisive choice. Below are possible cases one may encounter
        _bounding_interval, support, majority_label = self.compute_bounds(capture)
        if (self.configuration['accuracy_lowerbound'] and minimum_bounding_interval.lowerbound > support):
            # By Theorem 4, no possible choice would produce a subtree that could be part of a larger optimal subtree
            # So the optimal choice doesn't matter since this whole problem is irrelevant 
            result = Result(optimizer=None, optimum=Interval(float('Inf'))) # Makes parent minimization problems easier
            self.put(capture, path, result)
            self.prune(path[:-1])
            dependencies = tuple()
            if self.verbose or self.log:
                self.print('Case: Prune, Problem: {}:{} => {}'.format(path, capture, result))
        elif minimum_bounding_interval.uncertainty > 0:
            # The problem solution is still uncertain
            # We might be able to narrow the minimum_bounding_interval from the previous interval
            if self.configuration['interval_look_ahead']:
                result = Result(optimizer=None, optimum=minimum_bounding_interval, running=True)
            else:
                result = Result(optimizer=None, optimum=Interval(minimum_bounding_interval.lowerbound, float('Inf')), running=True)
            
            self.update(capture, path, result)

            # We might be able to prune subproblem paths that weren't previously pruned
            for j in irrelevant_splits:
                self.prune(path + (j,))

            # Be sure to re-enqueue this task since it's not finished
            if result.overwrites(current_result):
                self.enqueue(Task(self.prioritize(capture, path), capture, path))
            else:
                self.enqueue(Task(priority + self.configuration['deprioritization'], capture, path))

            if self.configuration['independence'] < 1 and (self.configuration['independence'] == 0 or random() >= self.configuration['independence']) and current_result.running:
                dependencies = tuple()
            else:
                dependencies = relevant_splits
            if self.verbose or self.log:
                self.print('Case: Downward, Problem: {}:{} => {}'.format(path, capture, result))
        elif minimum_bounding_interval.uncertainty == 0:
            # The uncertainty is fully resolved, so minimizing_split is the optimal choice
            if minimizing_split == None:
                optimizer = (None, majority_label)
            else:
                optimizer = (minimizing_split, None)
            result = Result(optimizer=optimizer, optimum=minimum_bounding_interval)
            result.count = count
            self.put(capture, path, result)
            self.prune(path)
            dependencies = tuple()
            if self.verbose or self.log:
                self.print('Case: Upward, Problem: {}:{} => {}'.format(path, capture, result))
        else:
            if self.verbose or self.log:
                self.print('Case: Undefined, Problem: {}:{} => {}'.format(path, capture, result))
            raise Exception("ParallelOSDTError: Undefined Case Reached")

        return dependencies

    def minimize_choice_of_split(self, capture, path):
        # This problem, concerning this capture, set has to make an choice between a leaf or one of j in 1:m feature splits
        # If a split is chosen, two subproblems are spawned and (eventually) optimized.

        # The computed upperbound is precicely the objective achieved with the leaf option is chosen
        bounding_interval, _support, _majority_label = self.compute_bounds(capture)

        # As each choice is considered, the best lowerbound and best upperbound are tracked
        # This gives the lowerbound and upperbound on the objective assuming we make the optimal choice between leaf or one of j in 1:m feature splits
        # The choice of split (None if choosing leaf) that minimizes the upperbound is tracked
        minimum_lowerbound = bounding_interval.upperbound
        minimum_upperbound = bounding_interval.upperbound
        minimizing_split = None
        # These default values imply that if no split is better than the leaf, the minimizing choice returned will represent the leaf

        # Keep a list of the lowerbound and upperbound for each j in 1:m splits to compare their potentials
        bounding_intervals = [None for _j in range(self.dataset.width)]

        # Each choice of splitting on feature j presents a left and right bisection of the capture set, which might need to be optimized
        for j, left_capture, right_capture in self.dataset.splits(capture):
            # Create the subproblem entry or load the existing one
            left_result = self.find_or_create_result(left_capture, path + (j, 'L'))
            left_lowerbound = left_result.optimum.lowerbound
            left_upperbound = left_result.optimum.upperbound

            # Create the subproblem entry or load the existing one
            right_result = self.find_or_create_result(right_capture, path + (j, 'R'))
            right_lowerbound = right_result.optimum.lowerbound
            right_upperbound = right_result.optimum.upperbound

            # Objectives are formulated so that adding the two subtree objectives result in objective of the larger tree formed
            # This provides the lowerbound and upperbound for splitting on feature j
            if self.configuration['hierarchical_lowerbound']:
                # The hierarchical bound lets us know that when subtrees are composed
                # The objective lowerbound of one subtree is irreducible by the other subtrees
                # Thus they can be simply added together, 
                split_lowerbound = left_lowerbound + right_lowerbound
            else:
                # In the absence of this theorem, we don't know how subtree objectives interact
                # So the best we can do is add the irreducible components
                left_bounding_interval, _support, _majority_label = self.compute_bounds(left_capture)
                right_bounding_interval, _support, _majority_label = self.compute_bounds(right_capture)
                split_lowerbound = min(
                    left_lowerbound + right_bounding_interval.lowerbound,
                    right_lowerbound + left_bounding_interval.lowerbound)
            
            split_upperbound = left_upperbound + right_upperbound
            bounding_interval = Interval(split_lowerbound, split_upperbound)

            # Store the bounding interval into a list for later comparison
            bounding_intervals[j] = bounding_interval

            # Update best 
            minimum_lowerbound = min(minimum_lowerbound, split_lowerbound)
            if split_upperbound < minimum_upperbound:
                minimum_upperbound = split_upperbound
                minimizing_split = j

        # The minimum_bounding_interval is the interval in which we know the minimum objective exists
        minimum_bounding_interval = Interval(minimum_lowerbound, minimum_upperbound)

        # Over the Leaf + M possible choices, each choice's minimum objective is bounded within an interval (bounding_intervals[j] or no interval for the leaf case)
        # I provides a threshold, such that choices with bounding intervals completely above the
        # minimum_bounding_interval need not be considered. These are placed in a separate partition labelled "irrelevant"
        # Additionally, both iterables exclude features already split in the path (history) of how this problem was reached
        if self.configuration['look_ahead']:
            # Lemma 2:
            # When look-ahead is enabled, we're able to compare candidates that minimize the same capture set and prune ones worse that the best so far
            # In this implementation, we don't have individual candidates but groups of candidates with a bounding interval over the group's optimal objective
            # So instead of using one best candidate to prune other candidates, we aggregate a best bounding interval and prune groups whose interval don't overlap

            relevant_splits = (j for j in self.dataset.gini_index if bounding_intervals[j].lowerbound <= minimum_bounding_interval.upperbound and not j in path)
            irrelevant_splits = (j for j in self.dataset.gini_index if bounding_intervals[j].lowerbound > minimum_bounding_interval.upperbound and not j in path) # SLOW
        else:
            # In the absence of look-ahead, we consider all possible splits worth considering even if the complexity cost of splitting
            # raises the lowerbound up to a level higher than possibly optimal
            relevant_splits = (j for j in self.dataset.gini_index if not j in path)
            irrelevant_splits = tuple()

        return minimum_bounding_interval, minimizing_split, tuple(relevant_splits), tuple(irrelevant_splits)

    def compute_bounds(self, capture):
        total, zeros, ones, minority, _majority = self.dataset.label_distribution(capture)
        # Lowerbound derived from equivalent points lowerbound on misclassification penalty with minimal regularization cost
        if self.configuration['equivalent_point_lowerbound']:
            lowerbound = minority / self.dataset.sample_size + self.lamb
        else:
            lowerbound = self.lamb
        # Upperbound derived from risk of a leaf over this set of captured points (should we look further?)
        upperbound = min(zeros, ones) / self.dataset.sample_size + self.lamb
        if not self.configuration['interval_look_ahead'] and lowerbound < upperbound:
            upperbound = float('Inf')

        if self.configuration['warm_start'] and capture.count() == len(capture):
            if self.initial_upperbound < upperbound:
                upperbound = min(upperbound, self.initial_upperbound)

        interval = Interval(lowerbound, upperbound)
        if self.configuration['similarity_threshold'] > 0:
            interval = self.results.converge(capture, Result(optimizer=None, optimum=interval)).optimum
        
        # Support
        support = total / self.dataset.sample_size
        majority_label = 0 if zeros >= ones else 1

        return interval, support, majority_label

    def is_pruned(self, path):
        if self.configuration['task_cancellation']:
            # or self.suffixes.shortest_prefix(path[::-1]) > 0
            return self.prefixes.shortest_prefix(path) > 0
        else:
            return False
        

    def prune(self, path):
        if self.configuration['task_cancellation']:
            if not self.is_pruned(path):
                self.prefixes.put(path, True)
            # if self.suffixes.shortest_prefix(path[::-1]) == 0:
            #     self.suffixes.put(path[::-1], True)
        pass

    def dequeue(self):
            task = self.tasks.pop(block=False)
            if task == None:
                return None
            if self.is_pruned(task.path):
                # self.print('Case: Pruned, Problem: {}:{}'.format(task.path, task.capture))
                return None
            return task
                    
    def enqueue(self, task):
        if not self.is_pruned(task.path):
            self.tasks.push(task, block=False)

    def get(self, capture, path):
        if self.configuration['capture_equivalence']:
            key = capture
        else:
            key = (capture, path)
        return self.results.get(key, block=False)

    def put(self, capture, path, result):
        if self.configuration['capture_equivalence']:
            key = capture
        else:
            key = (capture, path)
        self.results.put(key, result, block=False)

    def update(self, capture, path, result):
        if result.overwrites(self.get(capture, path)):
            self.put(capture, path, result)

    def solved(self, capture, path):
        result = self.get(capture, path)
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

    def timeout(self):
        return self.elapsed_time() > self.max_time

    # Method run by worker nodes to decide when to terminate
    def complete(self):
        # Termination condition
        # root = self.results.get(self.root, block=False)
        # terminate = root != None and root.optimizer != None and root.optimum.uncertainty == 0
        terminate = self.solved(self.root, tuple())
        return terminate

    # Method for extracting the output
    def output(self):
        return Tree(self.root, self.results, self.dataset, capture_equivalence=self.configuration['capture_equivalence'])

    def solve(self, workers=1, visualize_model=False, visualize_training=False):
        if self.verbose or self.log:
            self.print("Starting Parallel OSDT")

            self.print("Problem Definition:")
            self.print("  Number of Samples: {}".format(self.dataset.sample_size))
            self.print("  Number of Unique Samples: {}".format(self.dataset.height))
            self.print("  Number of Features: {}".format(self.dataset.width))
            self.print("  Regularization Coefficient: {}".format(self.lamb))

            self.print("Execetion Resources:")
            self.print("  Number of Woker Processes: {}".format(workers))
            self.print("  Number of Server Processes: {}".format(1))

            self.print("Algorithm Configurations:")
            for key, value in self.configuration.items():
                self.print("  {} = {}".format(key, value))
        
        self.visualize_training = visualize_training
        self.visualize_model = visualize_model
        self.workers = workers
        self.start_time = time()
        self.root = Vector.ones(self.dataset.height)  # Root capture
        cooldown = self.configuration['synchronization_cooldown']

        self.initial_upperbound = 1.0
        if self.configuration['warm_start']:
            # Warm start using cart
            cart = DecisionTreeClassifier(**{
                'max_depth': None,
                'min_samples_split': max(2, ceil(self.lamb * 2 * self.dataset.sample_size)),
                'min_samples_leaf': max(1, ceil(self.lamb * self.dataset.sample_size)),
                'max_leaf_nodes': max(2, floor(1 / (2 * self.lamb))),
                'min_impurity_decrease': self.lamb
            })
            cart.fit(self.dataset.original[0], self.dataset.original[1])
            loss = 1.0 - cart.score(self.dataset.original[0], self.dataset.original[1])
            width = compute_width(cart)
            cart_risk = loss + self.lamb * width
            self.initial_upperbound = cart_risk
            print("Initializing Uppbound as {}".format(self.initial_upperbound))
        else:
            print("Initializing Uppbound as {}".format(self.initial_upperbound))

        manager = Manager()

        # Set of "services" which are data structures that require management by a server process and get consumed by client processes
        prefixes = DictionaryService(
            table=PrefixTree(minimize=True), 
            degree=workers, 
            synchronization_cooldown=cooldown)

        if self.configuration['similarity_threshold'] > 0 and self.configuration['similarity_threshold'] != float('Inf'):
            false_negative_rate = 1 - 0.02 ** (self.configuration['similarity_threshold'] - 1) # / self.dataset.height
            max_tables = round(
                comb(self.dataset.height, self.configuration['similarity_threshold'])
            )
            tables = max(1, round( (1 - false_negative_rate) * max_tables ))
            print('height = {}; tables = {}'.format(self.dataset.height, tables))
            similarity_index = SimilarityIndex(
                distance=self.configuration['similarity_threshold'],
                dimensions=self.dataset.height,
                tables=tables)
            propagator = SimilarityPropagator(similarity_index, self.dataset, self.lamb, cooldown=cooldown)
        elif self.configuration['similarity_threshold'] == float('Inf'):
            print('Activating Full Index')
            similarity_index = FullIndex()
            propagator = SimilarityPropagator(similarity_index, self.dataset, self.lamb, cooldown=cooldown)
        else:
            propagator = None
        
        results = DictionaryService(
            table=ResultTable(table=dict()),
            degree=workers, 
            synchronization_cooldown=cooldown,
            propagator=propagator)
        results[0].dataset = self.dataset
        root_priority = 0
        tasks = QueueService(
            queue=HeapQueue( list([Task(root_priority, self.root, tuple())]) ),
            degree=workers,
            synchronization_cooldown=cooldown)
        
        services = (tasks, results, prefixes)

        # Initialize and run the cluster
        cluster = Cluster(self.task, services, size=workers, server_period=cooldown)
        model, duration = cluster.compute(self.max_time)

        if model != None:
            if self.verbose or self.log:
                self.print("Finishing Parallel OSDT in {} seconds".format(round(self.elapsed_time(), 3)))
                self.print("Optimal Objective: {}".format(model.risk))
            if self.visualize_model:
                model.visualize(self.dataset.width) # Renders a rule-list visualization
                if self.verbose or self.log:
                    self.print('Optimal Model:\n{}'.format(model.visualization))
            return model, duration
        else:
            if self.verbose or self.log:
                self.print("ParallelOSDTError: Early Termination after {} seconds".format(round(self.elapsed_time(), 3)))
            raise Exception("ParallelOSDTError: Early Termination after {} seconds".format(round(self.elapsed_time(), 3)))

    def __default_configuration__(self):
        return {
            'objective': 'accuracy', # Choose from accuracy, balanced_accuracy, weighted_accuracy
            'accuracy_weight': 1.0, # Only used for weighted accuracy

            'priority_metric': 'depth', # Decides how tasks are prioritized
            'deprioritization': 0.1, # Decides how much to push back a task if it has pending dependencies

            'warm_start': False, # Warm start with cart tree's risk as upperbound

            # Toggles the assumption about objective independence when composing subtrees (Theorem 1)
            'hierarchical_lowerbound': True, 
            # Toggles whether problems are pruned based on insufficient accuracy (compared to other results) (Lemma 2)
            'look_ahead': True,
            # Toggles whether a split is avoided based on insufficient support (proxy for accuracy gain) (Theorem 3)
            'support_lowerbound': True,
            # Toggles whether a split is avoided based on insufficient potential accuracy gain (Theorem 4)
            'incremental_accuracy_lowerbound': True,
            # Toggles whether a problem is pruned based on insufficient accuracy (in general) (Theorem 5)
            'accuracy_lowerbound': True,
            # Toggles whether problem equivalence is based solely on the capture set (Similar to Corollary 6)
            'capture_equivalence': True,
            # Hamming distance used to propagate bounding information of similar problems (Theorem 7 + some more...)
            "similarity_threshold": 0,
            # Toggles whether equivalent points contribute to the lowerbound (Proposition 8 and Theorem 9)
            'equivalent_point_lowerbound': True,

            # Toggles compression of dataset based on equivalent point aggregation
            'equivalent_point_compression': True,
            # Toggles whether asynchronous tasks can be cancelled after being issued
            'task_cancellation': False,
            # Toggles whether look_ahead prunes using objective upperbounds (This builds on top of look_ahead)
            'interval_look_ahead': True,
            # Cooldown timer (seconds) on synchornization operations
            'synchronization_cooldown': float('Inf'),
            # Cache Limit
            'independence': 1.0
        }

    def elapsed_time(self):
        return time() - self.start_time

    def print(self, message):
        # Internal print method to deal with verbosity and logging
        if self.verbose:
            print(message)
        if self.logger != None:
            self.logger.log([time(), message])


# def comb(N, k):  # from scipy.comb(), but MODIFIED!
#     if (k > N) or (N < 0) or (k < 0):
#         return 0L
#     N, k = map(long, (N, k))
#     top = N
#     val = 1L
#     while (top > (N-k)):
#         val *= top
#         top -= 1
#     n = 1L
#     while (n < k+1L):
#         val /= n
#         n += 1
#     return val
