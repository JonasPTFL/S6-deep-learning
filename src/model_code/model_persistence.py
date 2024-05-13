import tensorflow as tf
import src.util.constants as constants
import src.model_code.model_architecture as model_architecture

defaultModelSaveFilename = 'model_v1.keras'
defaultCheckpointModelSaveFilename = 'model_v1.weights.h5'


def model_load(filename: str = defaultModelSaveFilename) -> tf.keras.Model:
    """
    Loads the model from a file
    :return: the model
    """
    return tf.keras.models.load_model(constants.MODEL_DIRECTORY / filename)


def model_save(model: model_architecture.ModelArchitecture, save_file_name=defaultModelSaveFilename) -> None:
    """
    Saves the model to a file
    :param save_file_name: filename to save the model to
    :param model: the model to save
    """
    model.get_model().save(constants.MODEL_DIRECTORY / save_file_name)


def model_checkpoint_callback() -> tf.keras.callbacks.ModelCheckpoint:
    """
    Creates a model checkpoint callback, which saves the current model weights as checkpoint to a file
    :return:
    """
    # model checkpoint callback
    return tf.keras.callbacks.ModelCheckpoint(
        filepath=constants.CHECKPOINT_MODEL_DIRECTORY / defaultCheckpointModelSaveFilename,
        save_weights_only=True,  # save only the weights
        monitor='val_loss',  # monitor validation loss
        save_best_only=True,  # save only the best models
        mode='min',  # save the smallest validation loss
        verbose=1
    )
