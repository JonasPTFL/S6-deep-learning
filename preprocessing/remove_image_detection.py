import os
import shutil

import numpy as np
import constants
from preprocessing.data_utility import get_image_path_from_meta_data, get_image_path


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
    restore_keyword = 'restore'
    confirmation_input = input(
        f"Detected {removed_images_count} removed image(s). "
        f"Append them to file '{constants.REMOVED_IMAGES_STORAGE_PATH}' now? Type:\n"
        f"   '{confirmation_keyword}': append to file\n"
        f"   '{display_keyword}': show removed files\n"
        f"   '{restore_keyword}': restore removed files\n"
    )

    if confirmation_input == confirmation_keyword:
        # append removed images to file
        with open(constants.REMOVED_IMAGES_STORAGE_PATH, 'a') as removed_images_file:
            # sort removed images alphabetically
            removed_images = sorted(removed_images)
            for removed_image in removed_images:
                class_name = os.path.basename(os.path.dirname(removed_image))
                image_id = os.path.basename(removed_image).split('.')[0]
                removed_images_file.write(f"{class_name}/{image_id}\n")
    elif confirmation_input == display_keyword:
        # show removed images
        for removed_image in removed_images:
            print(removed_image)
    elif confirmation_input == restore_keyword:
        # restore removed images
        dataset_backup_path = input(
            "Enter the path to the backup of the dataset: (given path should contain folder: 'dataset/food-101/...')\n"
        )
        if dataset_backup_path:
            for removed_image in removed_images:
                # remove relative path prefix ("../") from removed_image when accessing the backup images
                backup_image_path = os.path.join(dataset_backup_path, removed_image[3:])
                # copy removed images from backup to original dataset
                shutil.copyfile(backup_image_path, removed_image)
else:
    print("No removed images detected.")
