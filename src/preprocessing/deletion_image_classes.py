from src.util import constants
import os
import shutil

chosen_classes = open(constants.CHOSEN_CLASSES_PATH).read().splitlines()


def delete_classes():
    all_dir_paths = []
    for directory_path, directory_name, filenames in os.walk(constants.ALL_IMAGES_PATH):
        all_dir_paths.append(directory_path)

    # filter the directories to remove
    for chosen_class in chosen_classes:
        for directory_path in all_dir_paths:
            if chosen_class in directory_path.split(os.path.sep):
                all_dir_paths.remove(directory_path)

    # remove first element, because it is the root folder of the images
    all_dir_paths = all_dir_paths[1:]

    # remove all directories and their files in them
    for i, entry in enumerate(all_dir_paths):
        # delete every file in the directory
        for filename in os.listdir(entry):
            file_path = os.path.join(entry, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))
        os.removedirs(entry)


delete_classes()
