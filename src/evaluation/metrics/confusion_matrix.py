from abc import ABC

import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from src.util import constants
import numpy as np


from src.evaluation.metrics.abstract_metric import AbstractMetric


def plot_confusion_matrix(y_true, y_pred, classes, model_id, timestamp,description):
    """
    This function plots the confusion matrix as pyplot plot.
    :param y_true: The true labels.
    :param y_pred: The predicted labels.
    :param classes: the class names of the classes.
    :param model_id: the model id of the model.
    :param timestamp: the timestamp of the model.
    :return:
    """
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=classes)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)

    fig, ax = plt.subplots(figsize=(25, 25))
    disp.plot(ax=ax)
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.savefig(f"../reports/{model_id}/{timestamp}/confusion_matrix_{description}.png")
    plt.clf()


class ConfusionMatrix(AbstractMetric, ABC):
    """
    This class implements the Confusion Matrix metric.
    """

    def calculate_metric(self, model: tf.keras.Model = None,
                         test_dataset: tf.data.Dataset = None,
                         train_dataset: tf.data.Dataset = None,
                         model_id: str = "-1", model_history=None,
                         model_timestamp=None):
        class_names = constants.CLASS_NAMES
        y_true = [label for images, labels in test_dataset for image, label in zip(images, labels)]
        y_true = np.argmax(y_true, axis=1)
        y_pred = model.predict(test_dataset).argmax(axis=1)
        plot_confusion_matrix(y_true, y_pred, class_names, model_id, model_timestamp, "test")
        plt.clf()
        y_true = [label for images, labels in train_dataset for image, label in zip(images, labels)]
        y_true = np.argmax(y_true, axis=1)
        y_pred = model.predict(train_dataset).argmax(axis=1)
        plot_confusion_matrix(y_true, y_pred, class_names, model_id, model_timestamp, "train")
        plt.clf()
