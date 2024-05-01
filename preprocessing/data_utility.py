import os

from preprocessing import constants


def get_image_path(class_name, image_id):
    """
    Creates the path to the image using the class name and the image id. Uses constant jpg as file extension since
    all images of the food-101 dataset are in jpg format.
    :param class_name: name of the class
    :param image_id: id of the image
    :return: project relative path to the image
    """
    # join path to dataset images with the class and id
    # ensure that the path uses forward slashes as join produces backslashes on windows
    return os.path.join(constants.all_images_classpath, class_name, f"{image_id}.jpg").replace("\\", "/")


def get_image_path_from_meta_data(metadata):
    """
    Extracts the class name and image id from the metadata and returns the path to the image.
    This metadata format is used in the meta files of the food-101 dataset.
    :param metadata: metadata of the image in the format 'class/id'
    :return: project relative path to the image
    """
    class_name, image_id = metadata.split("/")
    return get_image_path(class_name, image_id)
