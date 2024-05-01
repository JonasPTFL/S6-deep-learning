from src.util import constants
import os
import numpy as np
from src.util.data_utility import get_image_path

removed_images = set(np.loadtxt(constants.REMOVED_IMAGES_STORAGE_PATH, dtype=str))

# prompt user to remove the images
if input(f"Are you sure you want to delete {len(removed_images)} images? (yes/no)\n") == 'yes':
    # delete image files marked as removed
    for removed_image in removed_images:
        try:
            # extract class name and image id from format 'class/id'
            removed_image_split = removed_image.split('/')
            class_name = removed_image_split[0]
            image_id = removed_image_split[1]
            # get image path and delete the image
            image_path = get_image_path(class_name, image_id)
            os.remove(image_path)
            print(f'Deleted {removed_image}')
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (removed_image, e))
    print('Deletion operation completed.')
else:
    print('Delete operation canceled.')
