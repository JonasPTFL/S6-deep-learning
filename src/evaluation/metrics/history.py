from abc import ABC

from src.evaluation.metrics.abstract_metric import AbstractMetric
import tensorflow as tf
import matplotlib.pyplot as plt


def plot_loss_values(model_history):
    """
    Plot loss values of model history.
    :param model_history: the model history.
    :return:
    """
    plt.plot(model_history.history['loss'])
    plt.plot(model_history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()


def plot_accuracy_values(model_history):
    """
    Plot accuracy values of model history.
    :param model_history: the model history.
    :return:
    """
    plt.plot(model_history.history['accuracy'])
    plt.plot(model_history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()


class History(AbstractMetric, ABC):
    def calculate_metric(self, model: tf.keras.Model = None,
                         test_dataset: tf.data.Dataset = None,
                         model_id: str = "-1", model_history=None):
        plot_accuracy_values(model_history)
        plot_loss_values(model_history)
