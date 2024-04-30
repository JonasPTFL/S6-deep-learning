import random

import constants

class_names = []
chosen_classes = []

chosen_class_file = open(constants.chosen_classes_path, 'w')
with open(constants.all_available_classes_path, "r") as f:
    for line in f:
        class_names.append(line.strip())
    for i, class_name in enumerate(class_names):
        print(i, class_name)
    for i in range(0, constants.amount_classes):
        element = random.choice(class_names)
        chosen_classes.append(element)
        print(i, element)
        class_names.remove(element)
    chosen_class_file.write("\n".join(chosen_classes))
    chosen_class_file.close()
