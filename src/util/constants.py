from pathlib import Path

# get the base/root directory of the project, which is three directories up from the current directory, as this file is
# located in the folder 'src/util/'
BASE_DIR = Path(__file__).parent.parent.parent


def path_from_base(path: str) -> Path:
    """
    Returns a Path object from the base directory
    :param path: the path from the base directory
    :return: a Path object
    """
    return (BASE_DIR / path).resolve()


# dataset constants
NUM_CLASSES = 30
ALL_AVAILABLE_CLASSES_PATH = path_from_base('dataset/food-101/meta/classes.txt')
ALL_IMAGES_PATH = path_from_base('dataset/food-101/images/')

# meta data file paths
META_TEST_IMAGE_LIST_PATH = path_from_base('dataset/food-101/meta/test.txt')
META_TRAIN_IMAGES_LIST_PATH = path_from_base('dataset/food-101/meta/train.txt')

# path to the file containing the chosen classes
CHOSEN_CLASSES_PATH = path_from_base('chosen_classes.txt')

# path to storage file for images, that are removed during the data cleaning process
REMOVED_IMAGES_STORAGE_PATH = path_from_base('removed_images.txt')

# model
MODEL_PATH = path_from_base('models/saved_model/model_v1.keras')
CHECKPOINT_MODEL_PATH = path_from_base('models/checkpoints/model_v1.weights.h5')
