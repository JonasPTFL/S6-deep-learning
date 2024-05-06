import tensorflow as tf
import src.util.constants as constants


def model_load() -> tf.keras.Model:
    """
    Loads the model from a file
    :return: the model
    """
    return tf.keras.models.load_model(constants.MODEL_PATH)  # Load the saved model


def model_save(model) -> None:
    """
    Saves the model to a file
    :param model: the model to save
    """
    model.save(constants.MODEL_PATH)
