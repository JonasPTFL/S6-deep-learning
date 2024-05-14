import src.model_code.mode_iteration as mode_iteration
import tensorflow as tf

from src.model_code.model_architecture import ModelArchitecture
from src.util import constants

if __name__ == '__main__':
    # Print the gpus available (test for local development)
    print("GPUs available: ", tf.config.list_physical_devices('GPU'))

    # create model iteration
    model_iteration = mode_iteration.ModelIteration(
        model_architecture=ModelArchitecture(
            architecture=tf.keras.Sequential(
                [
                    tf.keras.layers.Rescaling(1. / 255),
                    tf.keras.layers.Conv2D(32, 3, activation='relu'),
                    tf.keras.layers.Conv2D(32, 3, activation='relu'),
                    tf.keras.layers.MaxPooling2D(),
                    tf.keras.layers.Conv2D(64, 3, activation='relu'),
                    tf.keras.layers.MaxPooling2D(),
                    tf.keras.layers.Conv2D(64, 3, activation='relu'),
                    tf.keras.layers.MaxPooling2D(),
                    tf.keras.layers.Conv2D(64, 3, activation='relu'),
                    tf.keras.layers.MaxPooling2D(),
                    tf.keras.layers.Conv2D(128, 3, activation='relu'),
                    tf.keras.layers.MaxPooling2D(),
                    tf.keras.layers.Flatten(),
                    tf.keras.layers.Dense(1024, activation='relu'),
                    tf.keras.layers.Dense(constants.NUM_CLASSES, activation='softmax')
                ]
            ),
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=["accuracy"]
        ),
        iteration_name='stoic_axolotl_6_2',
        epochs=10
    )
    # run model iteration
    model_iteration.run()
