import os
import numpy as np
import constants


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


def get_all_chosen_image_paths():
    """
    Extracts all image paths of the chosen classes from the meta files of the food-101 dataset.
    This may include paths to images that are removed during the data cleaning process.
    :return: list of all available image paths of the chosen classes
    """
    meta_test_image_list = np.loadtxt(constants.META_TEST_IMAGE_LIST_PATH, dtype=str)
    meta_train_images_list = np.loadtxt(constants.META_TRAIN_IMAGES_LIST_PATH, dtype=str)

    # all images in the dataset, includes classes not in chosen_classes and all currently removed images for data
    # cleaning difference between all_images and edited_image_paths results in the images that are removed
    all_images_metadata = np.concatenate((meta_test_image_list, meta_train_images_list))

    # split image metadata from the format 'class/id' to ['class', 'id']
    # this metadata format is used in the meta files of the food-101 dataset
    all_images_metadata_split = np.array([image_metadata.split("/") for image_metadata in all_images_metadata])

    # filter only chosen classes
    filtered_images_metadata = all_images_metadata[np.isin(all_images_metadata_split[:, 0], chosen_classes)]

    # extract id from every entry of the numpy array (format: 'class/id')
    return [get_image_path_from_meta_data(image_metadata) for image_metadata in filtered_images_metadata]


chosen_classes = np.loadtxt(constants.chosen_classes_path, dtype=str)

edited_image_paths = []
for root, dirs, files in os.walk(constants.all_images_classpath):
    current_class_dir_name = os.path.basename(root)
    if root != constants.all_images_classpath and current_class_dir_name in chosen_classes:
        for file in files:
            # get image path and append it to the edited_image_paths list
            # split the file name to remove the file extension
            edited_image_paths.append(get_image_path(current_class_dir_name, file.split('.')[0]))

# get the difference between all images and edited images which results in the removed images
all_chosen_image_paths = get_all_chosen_image_paths()
removed_images = list(set(all_chosen_image_paths) - set(edited_image_paths))
removed_images_count = len(removed_images)

if removed_images_count > 0:
    # prompt user to remove the images
    confirmation_keyword = 'yes'
    display_keyword = 'show'
    confirmation_input = input(
        f"Detected {removed_images_count} removed image(s). "
        f"Append them to file '{constants.REMOVED_IMAGES_STORAGE_PATH}' now? Type:\n"
        f"   '{confirmation_keyword}': append to file\n"
        f"   '{display_keyword}': show removed files\n"
    )

    if confirmation_input == confirmation_keyword:
        # append removed images to file
        with open(constants.REMOVED_IMAGES_STORAGE_PATH, 'a') as removed_images_file:
            for removed_image in removed_images:
                removed_images_file.write(f"{removed_image}\n")
    elif confirmation_input == display_keyword:
        # show removed images
        for removed_image in removed_images:
            print(removed_image)
