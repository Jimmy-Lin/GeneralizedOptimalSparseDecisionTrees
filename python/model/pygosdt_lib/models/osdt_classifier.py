# local imports
from lib.osdt import bbound

# This is mainly a wrapper class which makes the model compliant with Sci-kit Learn's DecisionTreeClassifier interface
# This allows for easy interoperability which (hopefully) would help with testing and acceptance into their library as well as
# adoption by other developers


class OSDTClassifier:
    """
    Model Interface for external interaction
    """

    def __init__(self,
                 # Regularization coefficient which effects the penalty on model complexity
                 regularization=0.1,

                 max_depth=float('Inf'),  # User-specified limit on the model
                 max_time=float('Inf'),  # User-specified limit on the runtime

                 clients=1,  # Parameter that varies based on how much computational resource is available
                 servers=1,  # Parameter that varies based on how much computational resource is available

                 # More configurations around toggling optimizations and prioritization options
                 configuration=None,
                 visualize=True,  # Toggle whether a rule-list visualization is rendered
                 verbose=True,  # Toggle whether event messages are printed
                 log=True):  # Toggle whether processes log their events

        self.model = None
        self.regularization = regularization

        self.max_depth = max_depth
        self.max_time = max_time

        if configuration != None:
            self.configuration = configuration 
        else:
            self.configuration = {
                'priority_metric': 'curiosity',
                'support_lowerbound': True,
                'incremental_accuracy_lowerbound': True,
                'accuracy_lowerbound': True,
                'equivalent_point_lowerbound': True,
                'look_ahead': True
            }

    def fit(self, X, y):
        self.model, self.width = bbound(X, y,
            self.regularization,
            prior_metric=self.configuration['priority_metric'],
            MAXDEPTH=self.max_depth,
            MAX_NLEAVES=float('Inf'),
            niter=float('Inf'),
            logon=False,
            support=self.configuration['support_lowerbound'],
            incre_support=self.configuration['incremental_accuracy_lowerbound'],
            accu_support=self.configuration['accuracy_lowerbound'],
            equiv_points=self.configuration['equivalent_point_lowerbound'],
            lookahead=self.configuration['look_ahead'],
            lenbound=True,
            R_c0=1,
            timelimit=self.max_time,
            init_cart=True,
            saveTree=False,
            readTree=False)
        return

    def predict(self, X_hat):
        if self.model == None:
            raise "Error: Model not yet trained"
        predictions, accuracy = self.model(X_hat)
        return predictions

    def score(self, X_hat, y_hat):
        if self.model == None:
            raise "Error: Model not yet trained"
        predictions, accuracy = self.model(X_hat, y_hat)
        return accuracy
