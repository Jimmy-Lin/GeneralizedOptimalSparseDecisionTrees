import time
from numpy import array

# local imports
from model.pygosdt_lib.models.parallel_osdt import ParallelOSDT

# This is mainly a wrapper class which makes the model compliant with Sci-kit Learn's DecisionTreeClassifier interface
# This allows for easy interoperability which (hopefully) would help with testing and acceptance into their library as well as
# adoption by other developers
class ParallelOSDTClassifier:
    """
    Model Interface for external interaction
    """
    def __init__(self, 
                regularization = 0.1, # Regularization coefficient which effects the penalty on model complexity

                max_depth = float('Inf'), # User-specified limit on the model
                max_time = float('Inf'), # User-specified limit on the runtime 

                workers = 1, # Parameter that varies based on how much computational resource is available

                configuration = None, # More configurations around toggling optimizations and prioritization options
                visualize_model=False,  # Toggle whether a rule-list visualization is rendered
                visualize_training=False,  # Toggle whether a dependency visualization is streamed
                verbose = False, # Toggle whether event messages are printed
                log = False,
                profile = False): # Toggle whether processes log their events

        self.model = None
        self.regularization = regularization

        self.max_depth = max_depth
        self.max_time = max_time

        self.workers = workers

        self.configuration = configuration
        self.visualize_model = visualize_model
        self.visualize_training = visualize_training
        self.verbose = verbose
        self.log = log
        self.profile = profile

    def fit(self, X, y):
        (n, m) = X.shape
        prestart = time.perf_counter()
        problem = ParallelOSDT(X, y, self.regularization,
            configuration=self.configuration,
            max_depth=self.max_depth, max_time=self.max_time,
            verbose=self.verbose, log=self.log, profile=self.profile)
        self.model, duration = problem.solve(workers=self.workers, visualize_model=self.visualize_model, visualize_training=self.visualize_training)
        if self.visualize_model:
            self.width = len(self.model.rule_lists(m))
        return duration

    # Make a prediction for the give unlablelled dataset
    def predict(self, X_hat):
        if self.model == None:
            raise "Error: Model not yet trained"
        (n, m) = X_hat.shape
        predictions = array( [ [self.model.predict(X_hat[i,:])] for i in range(n) ] )
        # predictions.reshape(n, 1)
        return predictions

    # Computes the model accuracy based on the input dataset
    # Depending on whether the input is the training set or testing set
    # This would either compute training accuracy or testing accuracy
    def score(self, X_hat, y_hat):
        if self.model == None:
            raise "Error: Model not yet trained"
        
        (n, m) = X_hat.shape
        predictions = self.predict(X_hat)
        accuracy = sum( int(predictions[i] == y_hat[i]) for i in range(n) ) / n
        self.accuracy = accuracy
        return accuracy

    def source(self):
        return self.model.source(self.regularization)
