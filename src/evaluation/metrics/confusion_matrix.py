import itertools
from abc import ABC

from src.evaluation.metrics.abstract_metric import AbstractMetric
import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


def plot_confusion_matrix(y_true, y_pred, classes, model_id, figsize=(15, 15)):
    plt.figure(figsize=figsize)
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, cmap="Blues", fmt='g', xticklabels=classes, yticklabels=classes)
    plt.title(f"Confusion Matrix of ModelId: {model_id}", fontweight='bold', fontsize=20)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    # Move x-axis labels to the top
    plt.tick_params(axis='x', which='both', bottom=False, top=True, labelbottom=False, labeltop=True, labelrotation=90)

    plt.show()


class ConfusionMatrix(AbstractMetric, ABC):
    def calculate_metric(self, model: tf.keras.Model = None, pass_test_data: tf.data.Dataset = None,
                         model_id: str = "-1"):
        class_names = pass_test_data.class_names
        y_true = np.concatenate([y for x, y in pass_test_data], axis=0)
        y_pred = model.predict(pass_test_data).argmax(axis=1)
        plot_confusion_matrix(y_true, y_pred, class_names, model_id)
