import pandas as pd
import numpy as np
from json import dumps, JSONEncoder
from numpy import array
from operator import add, eq, ge
from sklearn.metrics import confusion_matrix, accuracy_score, balanced_accuracy_score

class NumpyEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NumpyEncoder, self).default(obj)

class TreeClassifier:
    def __init__(self, source, encoder=None, X=None, y=None):
        self.source = source
        self.encoder = encoder
        if not X is None and not y is None:
            self.__initialize_loss__(X, y)

    def __initialize_loss__(self, X, y):
        (n, m) = X.shape
        for i in range(n):
            self.__initialize_sample_loss__(X.values[i,:], y.values[i,-1], 1/n, self.source)
        return

    def __initialize_sample_loss__(self, sample, label, weight, root):
        distribution = {}
        nodes = [root]
        while len(nodes) > 0:
            node = nodes.pop()
            if "prediction" in node:
                if node["prediction"] != label:
                    node["loss"] += weight
            else:
                value = sample[node["feature"]]
                reference = node["reference"]
                if node["relation"] == "==":
                    if value == reference:
                        nodes.append(node["true"])
                    else:
                        nodes.append(node["false"])
                elif node["relation"] == ">=":
                    if value >= reference:
                        nodes.append(node["true"])
                    else:
                        nodes.append(node["false"])
                elif node["relation"] == "<=":
                    if value <= reference:
                        nodes.append(node["true"])
                    else:
                        nodes.append(node["false"])
                elif node["relation"] == ">":
                    if value > reference:
                        nodes.append(node["true"])
                    else:
                        nodes.append(node["false"])
                elif node["relation"] == "<":
                    if value < reference:
                        nodes.append(node["true"])
                    else:
                        nodes.append(node["false"])
                else:
                    raise "Unsupported relational operator {}".format(node["relation"])

    def loss(self, root):
        loss_value = 0.0
        nodes = [root]
        while len(nodes) > 0:
            node = nodes.pop()
            if "prediction" in node:
                loss_value += node["loss"]
            else:
                nodes.append(node["true"])
                nodes.append(node["false"])
        return loss_value

    def classify(self, sample, root):
        distribution = {}
        nodes = [root]
        while len(nodes) > 0:
            node = nodes.pop()
            if "prediction" in node:
                return node["prediction"], 1 - node["loss"]
            else:
                value = sample[node["feature"]]
                reference = node["reference"]
                if node["relation"] == "==":
                    if value == reference:
                        nodes.append(node["true"])
                    else:
                        nodes.append(node["false"])
                elif node["relation"] == ">=":
                    if value >= reference:
                        nodes.append(node["true"])
                    else:
                        nodes.append(node["false"])
                elif node["relation"] == "<=":
                    if value <= reference:
                        nodes.append(node["true"])
                    else:
                        nodes.append(node["false"])
                elif node["relation"] == ">":
                    if value > reference:
                        nodes.append(node["true"])
                    else:
                        nodes.append(node["false"])
                elif node["relation"] == "<":
                    if value < reference:
                        nodes.append(node["true"])
                    else:
                        nodes.append(node["false"])
                else:
                    raise "Unsupported relational operator {}".format(node["relation"])

    def predict(self, X):
        if not self.encoder is None:
            X = pd.DataFrame(self.encoder.encode(X.values[:,:]), columns=self.encoder.headers)
        
        predictions = []
        (n, m) = X.shape
        for i in range(n):
            prediction, confidence = self.classify(X.values[i,:], self.source)
            predictions.append(prediction)
        return array(predictions)

    def confidence(self, X):
        if not self.encoder is None:
            X = pd.DataFrame(self.encoder.encode(X.values[:,:]), columns=self.encoder.headers)
        
        predictions = []
        (n, m) = X.shape
        for i in range(n):
            prediction, confidence = self.classify(X.values[i,:])
            predictions.append(confidence)
        return array(predictions)
        
    def error(self, X, y, weight=None):
        return 1 - self.score(X, y, weight=weight)

    def score(self, X, y, weight=None):
        y_hat = self.predict(X)
        if weight == "balanced":
            return balanced_accuracy_score(y, y_hat)
        else:
            return accuracy_score(y, y_hat, normalize=True, sample_weight=weight)

    def confusion(self, X, y, weight=None):
        return confusion_matrix(y_test, y_pred, sample_weight=weight)

    def groups_helper(self, node):
        if "prediction" in node:
            node["rules"] = {}
            groups = [node]
            return groups
        else:
            if "name" in node:
                name = node["name"]
            else:
                name = "feature_{}".format(node["feature"])
            reference = node["reference"]
            groups = []
            for condition_result in ["true", "false"]:
                subtree = node[condition_result]
                for group in self.groups_helper(subtree):

                    # For each group, add the corresponding rule
                    rules = group["rules"]
                    if not name in rules:
                        rules[name] = {}
                    rule = rules[name]
                    if node["relation"] == "==":
                        rule["type"] = "Categorical"
                        if "positive" not in rule:
                            rule["positive"] = set()
                        if "negative" not in rule:
                            rule["negative"] = set()
                        if condition_result == "true":
                            rule["positive"].add(reference)
                        elif condition_result == "false":
                            rule["negative"].add(reference)
                        else:
                            raise "OptimalSparseDecisionTree: Malformatted source {}".format(node)
                    elif node["relation"] == ">=":
                        rule["type"] = "Numerical"
                        if "max" not in rule:
                            rule["max"] = float("INF")
                        if "min" not in rule:
                            rule["min"] = -float("INF")
                        if condition_result == "true":
                            rule["min"] = max(reference, rule["min"])
                        elif condition_result == "false":
                            rule["max"] = min(reference, rule["max"])
                        else:
                            raise "OptimalSparseDecisionTree: Malformatted source {}".format(node)
                    else:
                        raise "Unsupported relational operator {}".format(node["relation"])
                    
                    # Add the modified group to the group list
                    groups.append(group)
            return groups    

    def groups(self):
        aggregated_groups = self.groups_helper(self.source)
        return aggregated_groups

    def __str__(self):
        cases = []
        for group in self.groups():
            predicates = []
            for name in sorted(group["rules"].keys()):
                domain = group["rules"][name]
                if domain["type"] == "Categorical":
                    if len(domain["positive"]) > 0:
                        predicates.append("{} = {}".format(name, list(domain["positive"])[0]))
                    elif len(domain["negative"]) > 0:
                        predicates.append("{} not in {{ {} }}".format(name, ", ".join([ str(v) for v in domain["negative"] ])) )
                    else:
                        raise "Invalid Rule"
                elif domain["type"] == "Numerical":
                    predicate = name
                    if domain["min"] != -float("INF"):
                        predicate = "{} <= ".format(domain["min"]) + predicate
                    if domain["max"] != float("INF"):
                        predicate = predicate + " < {}".format(domain["max"])
                    predicates.append(predicate)
            
            if len(predicates) == 0:
                condition = "if true then:"
            else:
                condition = "if {} then:".format(" and ".join(predicates))
            outcomes = []
            # for outcome, probability in group["distribution"].items():
            outcomes.append("    predicted {}: {}".format(group["name"], group["prediction"]))
            outcomes.append("    misclassification penalty: {}".format(round(group["loss"], 3)))
            outcomes.append("    complexity penalty: {}".format(round(group["complexity"], 3)))
            result = "\n".join(outcomes)
            cases.append("{}\n{}".format(condition, result))
        return "\n\nelse ".join(cases)
    
    def __repr__(self):
        return self.source

    def __latex__(self, node):
        if "prediction" in node:
            if "name" in node:
                name = node["name"]
            else:
                name = "feature_{}".format(node["feature"])
            return "[ ${}$ [ ${}$ ] ]".format(name, node["prediction"])
        else:
            if "name" in node:
                if "=" in node["name"]:
                    name = "{}".format(node["name"])
                else:
                    name = "{} {} {}".format(node["name"], node["relation"], node["reference"])
            else:
                name = "feature_{} {} {}".format(node["feature"], node["relation"], node["reference"])
            return "[ ${}$ {} {} ]".format(name, self.__latex__(node["true"]), self.__latex__(node["false"])).replace("==", " \eq ").replace(">=", " \ge ").replace("<=", " \le ")

    def latex(self):
        return self.__latex__(self.source)
    
    def json(self):
        return dumps(self.source, cls=NumpyEncoder)

    def leaves(self):
        leaves_counter = 0
        nodes = [self.source]
        while len(nodes) > 0:
            node = nodes.pop()
            if "prediction" in node:
                leaves_counter += 1
            else:
                nodes.append(node["true"])
                nodes.append(node["false"])
        return leaves_counter
    
    def nodes(self):
        nodes_counter = 0
        nodes = [self.source]
        while len(nodes) > 0:
            node = nodes.pop()
            if "prediction" in node:
                nodes_counter += 1
            else:
                nodes_counter += 1
                nodes.append(node["true"])
                nodes.append(node["false"])
        return nodes_counter

    def __len__(self):
        return self.leaves()

    def features(self):
        feature_set = set()
        nodes = [self.source]
        while len(nodes) > 0:
            node = nodes.pop()
            if "prediction" in node:
                continue
            else:
                feature_set.add(node["name"])
                nodes.append(node["true"])
                nodes.append(node["false"])
        return feature_set 

    def binary_features(self):
        return len(self.encoder.headers)

    def regularization_upperbound(self, X, y):
        if not self.encoder is None:
            X = pd.DataFrame(self.encoder.encode(X.values[:,:]), columns=self.encoder.headers)
        X.insert(X.shape[1], "class", y)
        regularization = 1.0
        min_acc_increase = self.minimum_accuracy_increase(X, self.source)
        min_leaf_accuracy = self.minimum_leaf_accuracy(self.source)
        max_depth = self.maximum_depth(self.source)
        leaf_count = self.leaves()
        # regularization = min(min_acc_increase, regularization)
        regularization = min(min_leaf_accuracy, regularization)
        regularization = min(1 / max_depth, regularization)
        regularization = min(1 / (2 * leaf_count), regularization)
        if regularization < 0:
            print("Inferring Regularizer: MAI: {}, MLA: {}, MD: {}, LC: {}".format(min_acc_increase, min_leaf_accuracy, max_depth, leaf_count))
        return regularization


    def maximum_depth(self, node=None):
        if node is None:
            node = self.source
        if "prediction" in node:
            return 1
        else:
            return 1 + max(self.maximum_depth(node["true"]), self.maximum_depth(node["false"]))


    def minimum_leaf_accuracy(self, node):
        if "prediction" in node:
            return 1 - node["loss"]
        else:
            return min(self.minimum_leaf_accuracy(node["true"]), self.minimum_leaf_accuracy(node["false"]))

    def minimum_accuracy_increase(self, dataset, node):
        distribution = {}
        accuracy_increase = 0
        if "prediction" in node:
            return 0
        else:
            (n,_) = dataset.shape
            distribution = dict()
            for i in range(n):
                target = dataset.iloc[i,-1]
                if not target in distribution:
                    distribution[target] = 1 / n
                else:
                    distribution[target] += 1 / n
            baseline_prediction = None
            baseline_likelihood = 0
            for target, likelihood in distribution.items():
                if likelihood > baseline_likelihood:
                    baseline_likelihood = likelihood
                    baseline_prediction = target

            split_likelihood = 1 - (self.loss(node["true"]) + self.loss(node["false"]))
            accuracy_increase = split_likelihood - baseline_likelihood
            # print("Baseline Accuracy: {}, Split Accuracy: {}".format(baseline_likelihood, split_likelihood))

            reference = node["reference"]
            if node["relation"] == "==":
                if not "prediction" in node["true"]:
                    subset = dataset[dataset[dataset.columns[node["feature"]]]==reference]
                    accuracy_increase = min(accuracy_increase, self.minimum_accuracy_increase(subset, node["true"]))
                if not "prediction" in node["false"]:
                    subset = dataset[dataset[dataset.columns[node["feature"]]]!=reference]
                    accuracy_increase = min(accuracy_increase, self.minimum_accuracy_increase(subset, node["false"]))
            elif node["relation"] == ">=":
                if not "prediction" in node["true"]:
                    subset = dataset[dataset[dataset.columns[node["feature"]]]>=reference]
                    accuracy_increase = min(accuracy_increase, self.minimum_accuracy_increase(subset, node["true"]))
                if not "prediction" in node["false"]:
                    subset = dataset[dataset[dataset.columns[node["feature"]]]<reference]
                    accuracy_increase = min(accuracy_increase, self.minimum_accuracy_increase(subset, node["false"]))
            elif node["relation"] == "<=":
                if not "prediction" in node["true"]:
                    subset = dataset[dataset[dataset.columns[node["feature"]]]<=reference]
                    accuracy_increase = min(accuracy_increase, self.minimum_accuracy_increase(subset, node["true"]))
                if not "prediction" in node["false"]:
                    subset = dataset[dataset[dataset.columns[node["feature"]]]>reference]
                    accuracy_increase = min(accuracy_increase, self.minimum_accuracy_increase(subset, node["false"]))
            elif node["relation"] == ">":
                if not "prediction" in node["true"]:
                    subset = dataset[dataset[dataset.columns[node["feature"]]]>reference]
                    accuracy_increase = min(accuracy_increase, self.minimum_accuracy_increase(subset, node["true"]))
                if not "prediction" in node["false"]:
                    subset = dataset[dataset[dataset.columns[node["feature"]]]<=reference]
                    accuracy_increase = min(accuracy_increase, self.minimum_accuracy_increase(subset, node["false"]))
            elif node["relation"] == "<":
                if not "prediction" in node["true"]:
                    subset = dataset[dataset[dataset.columns[node["feature"]]]<reference]
                    accuracy_increase = min(accuracy_increase, self.minimum_accuracy_increase(subset, node["true"]))
                if not "prediction" in node["false"]:
                    subset = dataset[dataset[dataset.columns[node["feature"]]]>=reference]
                    accuracy_increase = min(accuracy_increase, self.minimum_accuracy_increase(subset, node["false"]))
            else:
                raise "Unsupported relational operator {}".format(node["relation"])
        return accuracy_increase
