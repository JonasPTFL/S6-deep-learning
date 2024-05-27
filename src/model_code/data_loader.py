import tensorflow as tf

import src.util.constants as constants
import src.util.default_values as default_values


# DataLoader class, provides methods to load training and validation dataset
class DataLoader:
    def __init__(self, batch_size=default_values.batch_size, img_height=default_values.img_height, img_width=default_values.img_width,
                 seed=default_values.seed, validation_split=default_values.validation_split):
        """
        Constructor for DataLoader class. Sets parameters for loading dataset and uses default values if not provided.
        :param batch_size: the size of the batch
        :param img_height: the height of the image
        :param img_width: the width of the image
        :param seed: seed for random number generator for shuffling the datasets
        :param validation_split: the fraction of the dataset to use as validation data
        """
        self.batch_size = batch_size
        self.img_height = img_height
        self.img_width = img_width
        self.seed = seed
        self.validation_split = validation_split

    def load_training_data(self) -> tf.data.Dataset:
        """
        Loads training dataset from data directory with parameters set in constructor
        :return: training dataset
        """
        return self.load_data("training")

    def load_validation_data(self) -> tf.data.Dataset:
        """
        Loads validation dataset from data directory with parameters set in constructor
        :return: validation dataset
        """
        return self.load_data("validation")

    def load_data(self, subset: str) -> tf.data.Dataset:
        """
        Loads dataset from data directory with parameters set in constructor
        :param subset: the subset of the dataset to load
        :return: the dataset
        """
        current_dataset = tf.keras.utils.image_dataset_from_directory(
            constants.ALL_IMAGES_PATH,
            validation_split=self.validation_split,
            subset=subset,
            seed=self.seed,
            image_size=(self.img_height, self.img_width),
            batch_size=self.batch_size
        )

        normalization_layer = tf.keras.layers.Rescaling(1. / 255)
        normalized_dataset = current_dataset.map(lambda x, y: (normalization_layer(x), y))

        return normalized_dataset
