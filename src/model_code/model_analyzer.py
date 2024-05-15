import os

import tensorflow as tf
import src.model_code.model_architecture as ma
from src.evaluation.evaluator import Evaluator
from src.evaluation.metrics.confusion_matrix import ConfusionMatrix
from src.evaluation.metrics.false_classificated_images import FalseClassifiedImages
from src.evaluation.metrics.history import History
from src.evaluation.metrics.standard_evaluate import StandardEvaluate


def model_evaluate(
        model_architecture: ma.ModelArchitecture,
        history: tf.keras.callbacks.History,
        val_ds: tf.data.Dataset,
        model_id,
        timestamp
) -> str:
    """
    Displays the model architecture
    :param model_architecture: the model to display
    :param history: the history of the model returned by the training
    :param val_ds: the validation dataset
    :param model_id: the model id, or model iteration name
    :param timestamp: the timestamp when the model was created
    :return: the path to the directory where the evaluation reports are stored
    """
    model = model_architecture.get_model()
    model.summary()

    str_timestamp = str(timestamp).replace(':', '_')
    str_timestamp = str_timestamp.replace('.', '_')

    evaluator = Evaluator(
        model,
        val_ds,
        history,
        [
            ConfusionMatrix(),
            FalseClassifiedImages(),
            History(),
            StandardEvaluate(),
        ],
        model_id=model_id,
        timestamp=str_timestamp
    )

    path_to_dir = f"../reports/{model_id}/{str_timestamp}/"
    if not os.path.exists(path_to_dir):
        os.makedirs(path_to_dir)
        print(f"Created directory {path_to_dir}")
    else:
        print(f"Directory {path_to_dir} already exists")
    evaluator.evaluate()
    return path_to_dir
