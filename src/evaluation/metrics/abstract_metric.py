from abc import ABC, abstractmethod
import tensorflow as tf


class AbstractMetric(ABC):
    """
    Abstract class that defines the interface for all metrics
    that can be evaluated
    """

    @abstractmethod
    def calculate_metric(self, model: tf.keras.Model = None,
                         test_dataset: tf.data.Dataset = None,
                         model_id: str = "-1", model_history=None,
                         model_timestamp=None):
        """
        Abstract method to calculate metric of a deep learning model
        using image classification
        :param model_timestamp: The timestamp of the model when it started
        :param model_history: The history of the model if needed
        :param model_id: The id of the deep learning model if needed, else default value
        :param model: The model the metric needs to be evaluated on
        :param test_dataset: The test data the metric needs to be evaluated on
        :param model: The time the training has been started
        """
        pass
