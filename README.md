# S6-deep-learning

## Group Members
- [Jonas Pollpeter](https://github.com/JonasPTFL)
- [Julius Emil Arendt](https://github.com/Aremju)

## Preprocessing

### Preprocessing steps for the food-101 dataset
Link to **food-101** dataset: [https://www.kaggle.com/datasets/dansbecker/food-101](https://www.kaggle.com/datasets/dansbecker/food-101)

General information about generated files:
> [`chosen_classes.txt`](./chosen_classes.txt) is used to determine which classes to keep
> 
> [`removed_images.txt`](./removed_images.txt) is used to determine which images to delete

Steps for preprocessing:
1. download the dataset from the link above
2. unzip the dataset to directory `dataset`
3. path variables for the dataset paths are set in the [`constants.py`](./preprocessing/constants.py) file
4. [`class_selection.py`](./preprocessing/class_selection.py) is used to select a random subset of classes from the 
dataset and store the directory names of those classes as a list in [`chosen_classes.txt`](./chosen_classes.txt)
5. [`deletion_image_classes.py`](./preprocessing/deletion_image_classes.py) updates the local dataset by deleting all
classes that are removed from the dataset in class reduction process.
6. [`deletion_image_files.py`](./preprocessing/deletion_image_files.py) is used update the local dataset by deleting all
images that are removed from the dataset in data cleaning process