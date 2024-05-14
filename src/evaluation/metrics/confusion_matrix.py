from abc import ABC

import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import confusion_matrix

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
    plt.figure(figsize=(15, 15))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, cmap="Blues", fmt='g', xticklabels=classes, yticklabels=classes)
    plt.title(f"Confusion Matrix of ModelId: {model_id}", fontweight='bold', fontsize=20)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    # Move x-axis labels to the top
    plt.tick_params(axis='x', which='both', bottom=False, top=True, labelbottom=False, labeltop=True, labelrotation=90)

    plt.savefig(f"../reports/{model_id}/{timestamp}/confusion_matrix.png")
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
