from datetime import datetime

import src.util.constants as constants
import src.model_code.data_loader as dl
import src.model_code.model_analyzer as model_analyzer
import src.model_code.model_architecture as ma
import src.model_code.model_persistence as mp
import src.model_code.model_training as mt
import src.util.default_values as default_values
from src.util.architecture_to_markdown import model_architecture_to_markdown_table


class ModelIteration:
    def __init__(
            self,
            model_architecture: ma.ModelArchitecture = ma.ModelArchitecture(),
            data_loader: dl.DataLoader = dl.DataLoader(),
            train_ds=None,
            val_ds=None,
            epochs: int = None,
            iteration_name: str = default_values.model_iteration_name,
            allowed_to_run: bool = True,
            should_write_iteration_report: bool = True
    ):
        """
        Constructs model iteration with given model, training and validation dataset
        :param model_architecture: the model, if None, the default model is created
        :param train_ds: the training dataset, if None, the default dataset is loaded
        :param val_ds: the validation dataset, if None, the default dataset is loaded
        :param epochs: the number of epochs to train the model or None to use default value
        :param iteration_name: the name of the iteration (aka model_id)
        :param allowed_to_run: whether to allow model training or validation dataset
        :param should_write_iteration_report: whether to write the iteration report
        """
        self.data_loader = data_loader
        self.epochs = epochs
        self.iteration_name = iteration_name
        self.creation_timestamp = datetime.now()
        self.allowed_to_run = allowed_to_run
        self.should_write_iteration_report = should_write_iteration_report
        
        # load datasets or use given datasets if this instance is allowed to run
        if self.is_allowed_to_run():
            if train_ds is None or val_ds is None:
                self._load_datasets()
            else:
                self.train_ds = train_ds
                self.val_ds = val_ds

        # assign model
        self.model_architecture = model_architecture

    def is_allowed_to_run(self):
        return self.allowed_to_run

    def run(self) -> None:
        """
        Runs the model iteration
        """
        self._train()
        self._save()
        report_dir_path = self._evaluate()
        if self.should_write_iteration_report:
            self.save_iteration_report(report_dir_path)

        print(f'##########')
        print(f'########## Model iteration "{self.iteration_name}" finished')
        print(f'##########')

    def save_iteration_report(self, report_dir_path: str) -> None:
        """
        Saves the iteration report as a markdown file to the given reports directory.
        :param report_dir_path: path to the directory where the reports of this iteration are stored
        :return: None
        """
        saved_model_path = constants.MODEL_RELATIVE_PATH + self._get_save_filename()
        architecture_table = model_architecture_to_markdown_table(self.model_architecture.architecture)
        iteration_report = f"""
# Model Iteration {self.iteration_name}

## Iteration Summary
| Key | Value |
| --- | --- |
| Timestamp | {self.creation_timestamp} |
| Epochs | {self.epochs} |
| Data dimension: | {self.data_loader.img_width}x{self.data_loader.img_height} |
| Data batch size | {self.data_loader.batch_size} |
| Model saved as | {saved_model_path} |
| Allowed to run | {self.allowed_to_run} |

## Model Architecture
### Layers
{architecture_table}

### Parameters
| Key | Value |
| --- | --- |
| Optimizer | {self.model_architecture.optimizer.__class__.__name__} |
| Loss | {self.model_architecture.loss.__class__.__name__} |
| Metrics | {self.model_architecture.metrics} |
"""
        file = open(f'{report_dir_path}/{constants.MARKDOWN_REPORT_FILE_NAME}', 'w', encoding='utf-8')
        file.write(iteration_report)

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

    def _evaluate(self) -> str:
        """
        Evaluates the model
        :return: the path to the directory where the evaluation reports are stored
        """
        return model_analyzer.model_evaluate(self.model_architecture, self.history, self.val_ds,
                                             model_id=self.iteration_name,
                                             timestamp=self.creation_timestamp)
