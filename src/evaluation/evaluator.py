from abc import ABC


class Evaluator(ABC):
    def __init__(self, model, test_data, train_data, history, metrics, model_id, timestamp):
        """
        This class is used to evaluate a model against a set of test data.
        The user is able to pass in a list of metrics to evaluate against this specific
        set of data.
        :param model: the trained model that needs to be evaluated
        :param test_data: the test data that needs to be evaluated on
        :param history: the history of the model that needs to be evaluated
        :param metrics: the metrics that will be calculated against the test data
        :param model_id: the id of the model that needs to be evaluated, or just the iteration_name
        """
        self.model = model
        self.test_data = test_data
        self.metrics = metrics
        self.history = history
        self.model_id = model_id
        self.timestamp = timestamp
        self.train_data = train_data
        return

    def evaluate(self):
        """
        Evaluates the model against a set of test data
        and prints out the metrics calculated against the test data.
        :return: None
        """
        for metric in self.metrics:
            metric.calculate_metric(self.model,
                                    self.test_data,
                                    self.train_data,
                                    model_id=self.model_id,
                                    model_history=self.history,
                                    model_timestamp=self.timestamp)
        return
