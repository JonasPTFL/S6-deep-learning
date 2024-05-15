import tensorflow as tf
import src.model_code.model_persistence as model_persistence
import src.util.constants as constants
import src.model_code.model_architecture as model_architecture
import src.util.default_values as default_values


def model_train(
        model: model_architecture.ModelArchitecture,
        train_ds: tf.data.Dataset,
        val_ds: tf.data.Dataset,
        epochs: int = None) -> tf.keras.callbacks.History:
    """
    Trains the model with the given datasets
    :param model: the model to train
    :param train_ds: the training dataset
    :param val_ds: the validation dataset
    :param epochs: the number of epochs to train the model or None to use default value
    :return the model history
    """
    if epochs is None:
        epochs = default_values.epochs

    # add tensorboard callback
    tensorboard = tf.keras.callbacks.TensorBoard(log_dir=constants.LOG_DIR)

    return model.get_model().fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=[model_persistence.model_checkpoint_callback(), tensorboard]
    )
