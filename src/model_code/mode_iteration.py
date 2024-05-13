from datetime import datetime

import src.model_code.data_loader as dl
import src.model_code.model_analyzer as model_analyzer
import src.model_code.model_architecture as ma
import src.model_code.model_persistence as mp
import src.model_code.model_training as mt
import src.util.default_values as default_values


class ModelIteration:
    def __init__(
            self,
            model_architecture: ma.ModelArchitecture = ma.ModelArchitecture(),
            data_loader: dl.DataLoader = dl.DataLoader(),
            train_ds=None,
            val_ds=None,
            epochs: int = None,
            iteration_name: str = default_values.model_iteration_name
    ):
        """
        Constructs model iteration with given model, training and validation dataset
        :param model_architecture: the model, if None, the default model is created
        :param train_ds: the training dataset, if None, the default dataset is loaded
        :param val_ds: the validation dataset, if None, the default dataset is loaded
        :param epochs: the number of epochs to train the model or None to use default value
        """
        self.data_loader = data_loader
        self.epochs = epochs
        self.iteration_name = iteration_name
        self.creation_timestamp = datetime.now()
        # load datasets or use given datasets
        if train_ds is None or val_ds is None:
            self._load_datasets()
        else:
            self.train_ds = train_ds
            self.val_ds = val_ds

        # assign model
        self.model_architecture = model_architecture

    def run(self) -> None:
        """
        Runs the model iteration
        """
        self._train()
        self._save()
        self._evaluate()

        print(f'##########')
        print(f'########## Model iteration "{self.iteration_name}" finished')
        print(f'##########')

    def _get_save_filename(self) -> str:
        """
        Returns the filename for saving the model
        :return: the filename
        """
        # format timestamp as yyyymmdd_hhmmss
        timestamp_str = self.creation_timestamp.strftime('%Y%m%d_%H%M%S')
        return f'{self.iteration_name}_{timestamp_str}.keras'

    def _load_datasets(self) -> None:
        """
        Loads the datasets
        """
        self.train_ds = self.data_loader.load_training_data()
        self.val_ds = self.data_loader.load_validation_data()

    def _train(self) -> None:
        """
        Trains the model
        """
        self.history = mt.model_train(self.model_architecture, self.train_ds, self.val_ds, epochs=self.epochs)

    def _save(self) -> None:
        """
        Saves the model
        """
        mp.model_save(self.model_architecture, save_file_name=self._get_save_filename())

    def _evaluate(self) -> None:
        """
        Evaluates the model
        """
        model_analyzer.model_evaluate(self.model_architecture, self.history, self.val_ds)
