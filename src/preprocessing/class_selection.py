import random

from src.util import constants

class_names = []
chosen_classes = []

chosen_class_file = open(constants.CHOSEN_CLASSES_PATH, 'w')
with open(constants.ALL_AVAILABLE_CLASSES_PATH, "r") as f:
    for line in f:
        class_names.append(line.strip())
    for i in range(0, constants.NUM_CLASSES):
        element = random.choice(class_names)
        chosen_classes.append(element)
        class_names.remove(element)
    chosen_class_file.write("\n".join(chosen_classes))
    chosen_class_file.close()
