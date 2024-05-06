import tensorflow as tf

from src.util import constants


def model_checkpoint_callback() -> tf.keras.callbacks.ModelCheckpoint:
    """
    Creates a model checkpoint callback, which saves the current model weights as checkpoint to a file
    :return:
    """
    # model checkpoint callback
    return tf.keras.callbacks.ModelCheckpoint(
        filepath=constants.CHECKPOINT_MODEL_PATH,
        save_weights_only=True,  # save only the weights
        monitor='val_loss',  # monitor validation loss
        save_best_only=True,  # save only the best models
        mode='min',  # save the smallest validation loss
        verbose=1
    )


def model_train(model, train_ds, val_ds) -> None:
    """
    Trains the model with the given datasets
    :param model: the model to train
    :param train_ds: the training dataset
    :param val_ds: the validation dataset
    """
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=3,
        callbacks=[model_checkpoint_callback()]
    )
