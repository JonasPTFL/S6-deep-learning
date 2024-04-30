import constants
import os
import numpy as np

removed_images = set(np.loadtxt(constants.REMOVED_IMAGES_STORAGE_PATH, dtype=str))

# prompt user to remove the images
if input(f"Are you sure you want to delete {len(removed_images)} images? (yes/no)\n") == 'yes':
    # delete image files marked as removed
    for removed_image in removed_images:
        try:
            os.remove(removed_image)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (removed_image, e))
    print('Deletion operation completed.')
else:
    print('Delete operation canceled.')
