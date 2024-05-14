from abc import ABC

from src.evaluation.metrics.abstract_metric import AbstractMetric
import tensorflow as tf


class StandardEvaluate(AbstractMetric, ABC):
    """
    This class implements a standard evaluation metric using TensorFlow
    model.evaluate() to print the compiled metrics.
    """
    def calculate_metric(self, model: tf.keras.Model = None,
                         test_dataset: tf.data.Dataset = None,
                         model_id: str = "-1", model_history=None,
                         model_timestamp=None):
        print("Evaluating model with Model ID {}".format(model_id))
        evaluation = model.evaluate(test_dataset)
        print("Loss: ", evaluation[0])
        print("Accuracy: ", evaluation[1])
