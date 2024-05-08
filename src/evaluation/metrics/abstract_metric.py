from abc import ABC, abstractmethod
import tensorflow as tf


class AbstractMetric(ABC):
    """
    Abstract class that defines the interface for all metrics
    that can be evaluated
    """

    @abstractmethod
    def calculate_metric(self, model: tf.keras.Model, test_data: tf.data.Dataset, model_id: str = "-1"):
        """
        Abstract method to calculate metric of a deep learning model
        using image classification
        :param model_id: The id of the deep learning model if needed, else default value
        :param model: The model the metric needs to be evaluated on
        :param test_data: The test data the metric needs to be evaluated on
        """
        pass
