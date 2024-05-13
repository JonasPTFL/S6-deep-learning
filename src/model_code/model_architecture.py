import tensorflow as tf

from src.util import constants

default_model_architecture = tf.keras.Sequential([
    tf.keras.layers.Rescaling(1. / 255),
    tf.keras.layers.Conv2D(32, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(32, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(32, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(constants.NUM_CLASSES, activation='softmax')
])
default_compile_metrics = ['accuracy']
default_loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True)
default_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)


class ModelArchitecture:
    def __init__(
            self,
            architecture: tf.keras.Sequential = None,
            optimizer=default_optimizer,
            loss=default_loss,
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
            self.architecture = default_model_architecture
        else:
            self.architecture = architecture

        if metrics is None:
            metrics = default_compile_metrics

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
