import tensorflow as tf

from src.util import constants

# default values for DataLoader class
default_batch_size = 32
default_img_height = 256
default_img_width = 256
default_seed = 123
default_validation_split = 0.2


# DataLoader class, provides methods to load training and validation dataset
class DataLoader:
    def __init__(self, batch_size=default_batch_size, img_height=default_img_height, img_width=default_img_width,
                 seed=default_seed, validation_split=default_validation_split):
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
        return tf.keras.utils.image_dataset_from_directory(
            constants.ALL_IMAGES_PATH,
            validation_split=self.validation_split,
            subset="training",
            seed=self.seed,
            image_size=(self.img_height, self.img_width),
            batch_size=self.batch_size
        )

    def load_validation_data(self) -> tf.data.Dataset:
        """
        Loads validation dataset from data directory with parameters set in constructor
        :return: validation dataset
        """
        return tf.keras.utils.image_dataset_from_directory(
            constants.ALL_IMAGES_PATH,
            validation_split=self.validation_split,
            subset="validation",
            seed=self.seed,
            image_size=(self.img_height, self.img_width),
            batch_size=self.batch_size
        )
