from abc import ABC
from typing import List

from metrics.abstract_metric import AbstractMetric


class Evaluator(ABC):
    def __init__(self, model, test_data, metrics: List[AbstractMetric]):
        """
        This class is used to evaluate a model against a set of test data.
        The user is able to pass in a list of metrics to evaluate against this specific
        set of data.
        :param model: the trained model that needs to be evaluated
        :param test_data: the test data that needs to be evaluated on
        :param metrics: the metrics that will be calculated against the test data
        """
        self.model = model
        self.test_data = test_data
        self.metrics = metrics
        return

    def evaluate(self):
        """
        Evaluates the model against a set of test data
        and prints out the metrics calculated against the test data.
        :return: None
        """
        for metric in self.metrics:
            metric.calculate_metric(self.model, self.test_data)
        return
