class Tree:
    def __init__(self, capture, table, dataset, path=None, capture_equivalence=True):
        if path == None:
            path = tuple()
        if capture_equivalence:
            key = capture
        else:
            key = (capture, path)
        if not table.has(key):
            print("Insufficent information for key {}".format(key))
        result = table.get(key, block=False)
        if result.optimizer == None:
            print("Insufficent information for key {}".format(key))
        (split, prediction) = result.optimizer

        risk = result.optimum.value()
        self.risk = risk
        self.split = split
        self.prediction = prediction
        self.visualization = None
        if split != None:
            (left_capture, right_capture) = dataset.split(split, capture=capture)
            self.left_subtree = Tree(left_capture, table, dataset, path=path + (split, 'L'), capture_equivalence=capture_equivalence)
            self.right_subtree = Tree(right_capture, table, dataset, path=path + (split, 'R'), capture_equivalence=capture_equivalence)

    def predict(self, sample):
        if self.prediction != None:
            return self.prediction
        elif sample[self.split] == 0:
            return self.left_subtree.predict(sample)
        elif sample[self.split] == 1:
            return self.right_subtree.predict(sample)

    def rule_lists(self, dimensions):
        if self.prediction != None:
            rule_lists = (
                (
                    ('_',) * dimensions,
                    self.prediction,
                    self.risk
                ),
            )
            return rule_lists
        else:
            left_rule_lists = (
                (
                    tuple('0' if j == self.split else rule_list[0][j] for j in range(dimensions)), 
                    *rule_list[1:]
                ) for rule_list in self.left_subtree.rule_lists(dimensions))
            right_rule_lists = (
                (
                    tuple('1' if j == self.split else rule_list[0][j] for j in range(dimensions)),
                    *rule_list[1:]
                ) for rule_list in self.right_subtree.rule_lists(dimensions))
            return tuple(left_rule_lists) + tuple(right_rule_lists)

    def visualize(self, dimensions):
        if self.visualization == None:
            visualization = '\n'.join("({}) => {} (Risk Contribution = {})".format(','.join(rule_list[0]), rule_list[1], rule_list[2]) for rule_list in self.rule_lists(dimensions))
            self.visualization = visualization
        return self.visualization

    def source(self, lamb):
        if self.prediction != None:
            return {
                "complexity": lamb,
                "loss": self.risk - lamb,
                "name": "class",
                "prediction": self.prediction
            }
        else:
            return {
                "feature": self.split,
                "name": "feature_{}".format(self.split),
                "reference": 1,
                "relation": "==",
                "true": self.right_subtree.source(lamb),
                "false": self.left_subtree.source(lamb),
            }

