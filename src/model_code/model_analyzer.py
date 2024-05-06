import tensorflow as tf


def model_evaluate(model: tf.keras.Model) -> None:
    """
    Displays the model architecture
    :param model: the model to display
    """
    model.summary()
