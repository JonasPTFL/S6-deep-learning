import random
from abc import ABC

import tensorflow as tf
from matplotlib import pyplot as plt

from src.evaluation.metrics.abstract_metric import AbstractMetric


def filter_and_plot(dataset, pred_labels, model_id):
    """
    Filters out correct classified images and plots
    a random set of false classified images
    :param model_id:
    :param dataset: The dataset to evaluate
    :param pred_labels: The predicted labels
    """
    false_images = filter_out_correct_classified_images(dataset, pred_labels)
    plot_false_images(false_images, model_id)


def plot_false_images(false_images, model_id):
    """
    Plots false images using matplotlib
    :param model_id:
    :param false_images: list of triples with false classified images, its true label and its predicted label
    """
    for count in range(10):
        # Ensure we only plot up to 18 random images
        random.shuffle(false_images)
        false_images = false_images[:18]

        num_images = len(false_images)
        num_cols = 6  # Number of columns in the grid
        num_rows = -(-num_images // num_cols)  # Equivalent to math.ceil(num_images / num_cols)
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(30, 10))  # Adjust size as needed
        for i, (image, pred_class, true_class) in enumerate(false_images):
            ax = axes[i // num_cols, i % num_cols] if num_rows > 1 else axes[i % num_cols]
            ax.imshow(image)
            ax.set_title(f"Predicted: {pred_class}, True: {true_class}")
            ax.axis('off')
        plt.suptitle(f"False Classified Images (Model ID: {model_id}), (Iteration: {count})", fontsize=30, fontweight='bold')
        plt.tight_layout()
        plt.show()


def filter_out_correct_classified_images(dataset, pred_labels):
    """
    Removes true classified images.
    :param dataset: the dataset to evaluate
    :param pred_labels: the predicted labels
    :return: A list of false classified images, its true label and its predicted label as triples
    """
    index = 0
    class_names = dataset.class_names
    false_images = []  # Accumulate false images here
    for images, labels in dataset.take(len(dataset)):
        for image, true_label in zip(images, labels):
            if true_label != pred_labels[index]:
                false_images.append((image.numpy().astype("uint8"), class_names[pred_labels[index]],
                                     class_names[true_label]))
            index += 1
    return false_images


class FalseClassifiedImages(AbstractMetric, ABC):
    """
    The false classified images metric.
    """
    def calculate_metric(self, model: tf.keras.Model, test_dataset: tf.data.Dataset, model_id: str = "-1"):
        y_pred_labels = model.predict(test_dataset).argmax(axis=1)
        filter_and_plot(test_dataset, y_pred_labels, model_id)
        return
