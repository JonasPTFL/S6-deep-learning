from abc import ABC

import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


from src.evaluation.metrics.abstract_metric import AbstractMetric


def plot_confusion_matrix(y_true, y_pred, classes, model_id, timestamp):
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

    # Anpassen der Figurengröße und Rotieren der Labels
    fig, ax = plt.subplots(figsize=(25, 25))  # Die Figurgröße kann nach Bedarf angepasst werden
    disp.plot(ax=ax)
    plt.xticks(rotation=90)  # Drehen der xtick Labels um 90 Grad
    plt.yticks(rotation=0)  # Drehen der ytick Labels um 0 Grad (optional, da es standardmäßig so ist)
    plt.show()


class ConfusionMatrix(AbstractMetric, ABC):
    """
    This class implements the Confusion Matrix metric.
    """

    def calculate_metric(self, model: tf.keras.Model = None,
                         test_dataset: tf.data.Dataset = None,
                         model_id: str = "-1", model_history=None,
                         model_timestamp=None):
        class_names = test_dataset.class_names
        y_true = [label for images, labels in test_dataset for image, label in zip(images, labels)]
        y_pred = model.predict(test_dataset).argmax(axis=1)
        plot_confusion_matrix(y_true, y_pred, class_names, model_id, model_timestamp)

