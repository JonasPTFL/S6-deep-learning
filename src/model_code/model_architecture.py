import tensorflow as tf

import src.util.default_values as default_values


class ModelArchitecture:
    def __init__(
            self,
            architecture: tf.keras.Sequential = None,
            optimizer=default_values.optimizer,
            loss=default_values.loss,
            metrics=None
    ):
        """
        Constructs model with given architecture or default architecture if none is provided and compiles it
        :param architecture: the architecture of the model, if None, the default architecture is used
        :param optimizer: the optimizer for compiling the model
        :param loss: the loss function for compiling the model
        :param metrics: the metrics to evaluate the model, uses default metrics if none are provided
        """
        if architecture is None:
            self.architecture = default_values.model_architecture
        else:
            self.architecture = architecture

        if metrics is None:
            metrics = default_values.compile_metrics

        # compile the model
        self.architecture.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics
        )

    def get_model(self) -> tf.keras.Sequential:
        """
        :return: the compiled model
        """
        return self.architecture
