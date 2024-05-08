from abc import ABC, abstractmethod


class AbstractMetric(ABC):
    """
    Abstract class that defines the interface for all metrics
    that can be evaluated
    """
    @abstractmethod
    def calculate_metric(self, model, test_data):
        """
        Abstract method to calculate metric of a deep learning model
        using image classification
        :param model: The model the metric needs to be evaluated on
        :param test_data: The test data the metric needs to be evaluated on
        """
        pass
