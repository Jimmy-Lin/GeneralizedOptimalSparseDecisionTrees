import pandas as pd
import numpy as np
from cplex import Cplex
from cplex.callbacks import MIPInfoCallback

class StatsCallback(MIPInfoCallback):

    def initialize(self):

        # scalars
        self.times_called = 0
        self.start_time = None

        # stats that are stored at every call len(stat) = times_called
        self.runtimes = []
        self.nodes_processed = []
        self.nodes_remaining = []
        self.lowerbounds = []

        # stats that are stored at every incumbent update
        self.best_objval = float('inf')
        self.update_iterations = []
        self.upperbounds = []
        self.process_incumbent = self.record_objval_before_incumbent


    def __call__(self):
        """
        this function is called everytime
        :return:
        """

        self.times_called += 1

        if self.start_time is None:
            self.start_time = self.get_start_time()

        self.runtimes.append(self.get_time())
        self.lowerbounds.append(self.get_best_objective_value())
        self.nodes_processed.append(self.get_num_nodes())
        self.nodes_remaining.append(self.get_num_remaining_nodes())
        self.process_incumbent()

    def record_objval_before_incumbent(self):
        if self.has_incumbent():
            self.record_objval()
            self.process_incumbent = self.record_objval

    def record_objval_and_solution_before_incumbent(self):
        if self.has_incumbent():
            self.record_objval_and_solution()
            self.process_incumbent = self.record_objval_and_solution

    def record_objval(self):
        objval = self.get_incumbent_objective_value()
        if objval < self.best_objval:
            self.best_objval = objval
            self.update_iterations.append(self.times_called)
            self.upperbounds.append(objval)

    def check_stats(self):
        """checks stats rep at any point during the solution process"""

        # try:
        n_calls = len(self.runtimes)
        n_updates = len(self.upperbounds)
        assert n_updates <= n_calls

        if n_calls > 0:

            assert len(self.nodes_processed) == n_calls
            assert len(self.nodes_remaining) == n_calls
            assert len(self.lowerbounds) == n_calls

            lowerbounds = np.array(self.lowerbounds)
            for ub in self.upperbounds:
                pass #assert np.all(ub > lowerbounds)

            runtimes = np.array(self.runtimes) - self.start_time
            nodes_processed = np.array(self.nodes_processed)

            is_increasing = lambda x: (np.diff(x) >= 0).all()
            assert is_increasing(runtimes)
            assert is_increasing(nodes_processed)
            #assert is_increasing(lowerbounds)

        if n_updates > 0:

            assert len(self.update_iterations) == n_updates
            update_iterations = np.array(self.update_iterations)
            upperbounds = np.array(self.upperbounds)
            gaps = (upperbounds - lowerbounds[update_iterations - 1]) / (np.finfo(np.float).eps + upperbounds)

            is_increasing = lambda x: (np.diff(x) >= 0).all()
            assert is_increasing(update_iterations)
            assert is_increasing(-upperbounds)
            assert is_increasing(-gaps)

        # except AssertionError:
        #    ipsh()

        return True

    def get_stats(self):
        """
        :return: DataFrame of MIP Statistics
        """

        assert self.check_stats()
        import pandas as pd
        MAX_UPPERBOUND = float('inf')
        MAX_GAP = 1.00

        stats = pd.DataFrame({
            'runtime': [t - self.start_time for t in self.runtimes],
            'nodes_processed': list(self.nodes_processed),
            'nodes_remaining': list(self.nodes_remaining),
            'lowerbound': list(self.lowerbounds)
            })

        upperbounds = list(self.upperbounds)
        update_iterations = list(self.update_iterations)

        # add upper bounds as well as iterations where the incumbent changes
        if update_iterations[0] > 1:
            update_iterations.insert(0, 1)
            upperbounds.insert(0, MAX_UPPERBOUND)
        row_idx = [i - 1 for i in update_iterations]
        stats = stats.assign(iterations = pd.Series(data = update_iterations, index = row_idx),
                             upperbound = pd.Series(data = upperbounds, index = row_idx))

        stats['incumbent_update'] = np.where(~np.isnan(stats['iterations']), True, False)
        stats = stats.fillna(method = 'ffill')

        # add relative gap
        gap = (stats['upperbound'] - stats['lowerbound']) / (stats['upperbound'] + np.finfo(np.float).eps)
        stats['gap'] = np.fmin(MAX_GAP, gap)

        # add model ids
        return stats[['runtime', 'gap', 'upperbound', 'lowerbound', 'nodes_processed', 'nodes_remaining']]


#stats['runtime'] <- time
#stats['nodes_processed'] <- # of BB nodes processed
#stats['nodes_remaining'] <- # of BB nodes remaining
#stats['lowerbound'] <- lowerbound at each time step
#stats['upperbound'] <- upperbound at each time step

