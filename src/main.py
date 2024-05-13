import model_code.data_loader as data_loader
import model_code.model_architecture as model_architecture
import model_code.model_training as model_training
import model_code.model_analyzer as analyze_model
import model_code.model_persistence as model_persistence
from src.evaluation.evaluator import Evaluator
from src.evaluation.metrics.confusion_matrix import ConfusionMatrix
from src.evaluation.metrics.false_classificated_images import FalseClassifiedImages
from src.evaluation.metrics.history import History
from src.evaluation.metrics.standard_evaluate import StandardEvaluate
import tensorflow as tf

if __name__ == '__main__':
    # load data
    data_loader = data_loader.DataLoader(img_height=224, img_width=224)
    train_ds = data_loader.load_training_data()
    val_ds = data_loader.load_validation_data()
    # Print the gpus available (test for local development)
    print("GPUs available: ", tf.config.list_physical_devices('GPU'))

    # create model
    model = model_architecture.model_architecture()

    # train model
    history = model_training.model_train(model, train_ds, val_ds)

    # save model
    model_persistence.model_save(model)

    # evaluate model
    analyze_model.model_evaluate(model)

    evaluator = Evaluator(
        model,
        val_ds,
        history,
        [
            ConfusionMatrix(),
            FalseClassifiedImages(),
            History(),
            StandardEvaluate(),
        ]
    )

    evaluator.evaluate()
