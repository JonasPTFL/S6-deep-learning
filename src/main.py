import os
import sys

sys.path.append(os.getcwd())

from src.model_code.mode_iteration import ModelIteration
import tensorflow as tf

from src.model_code.model_architecture import ModelArchitecture
from src.util import constants

transfer_learning_model = tf.keras.Sequential()
transfer_learning_model_1 = tf.keras.Sequential()

resnet50 = tf.keras.applications.ResNet50(
    include_top=False,
    weights="imagenet",
    input_shape=(224, 224, 3),
    pooling=None
)

for layer in resnet50.layers:
    layer.trainable = False

transfer_learning_model.add(resnet50)
transfer_learning_model.add(tf.keras.layers.Flatten())
transfer_learning_model.add(tf.keras.layers.Dense(1024, activation='relu'))
transfer_learning_model.add(tf.keras.layers.Dense(constants.NUM_CLASSES, activation='softmax'))

vgg16 = tf.keras.applications.VGG16(include_top=False,
                                    weights="imagenet",
                                    input_shape=(224,224,3),
                                    pooling=None)

for layer in vgg16.layers:
    layer.trainable = False

transfer_learning_model_1.add(vgg16)
transfer_learning_model_1.add(tf.keras.layers.Flatten())
transfer_learning_model_1.add(tf.keras.layers.Dense(1024, activation="relu"))
transfer_learning_model_1.add(tf.keras.layers.Dense(constants.NUM_CLASSES, activation="softmax"))


transfer_learning_model_2 = tf.keras.Sequential()

vgg19 = tf.keras.applications.VGG19(include_top=False,
                                    weights="imagenet",
                                    input_shape=(224,224,3),
                                    pooling=None)

for layer in vgg19.layers:
    layer.trainable = False

transfer_learning_model_2.add(vgg19)
transfer_learning_model_2.add(tf.keras.layers.Flatten())
transfer_learning_model_2.add(tf.keras.layers.Dense(1024, activation="relu"))
transfer_learning_model_2.add(tf.keras.layers.Dense(constants.NUM_CLASSES, activation="softmax"))


transfer_learning_model_3 = tf.keras.Sequential()

resnet101 = tf.keras.applications.ResNet101(include_top=False,
                                    weights="imagenet",
                                    input_shape=(224,224,3),
                                    pooling=None)

for layer in resnet101.layers:
    layer.trainable = False

transfer_learning_model_3.add(resnet101)
transfer_learning_model_3.add(tf.keras.layers.Flatten())
transfer_learning_model_3.add(tf.keras.layers.Dense(1024, activation="relu"))
transfer_learning_model_3.add(tf.keras.layers.Dense(constants.NUM_CLASSES, activation="softmax"))

model_iterations = [
    ModelIteration(
        model_architecture=ModelArchitecture(
            architecture=tf.keras.Sequential(
                [
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
        epochs=100,
        allowed_to_run=False
    ),
    ModelIteration(
        model_architecture=ModelArchitecture(
            architecture=tf.keras.Sequential(
                [
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
                    tf.keras.layers.Conv2D(128, 3, activation='relu'),
                    tf.keras.layers.MaxPooling2D(),
                    tf.keras.layers.Flatten(),
                    tf.keras.layers.Dense(1024, activation='relu'),
                    tf.keras.layers.Dropout(0.2),
                    tf.keras.layers.Dense(constants.NUM_CLASSES, activation='softmax')
                ]
            ),
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=["accuracy"]
        ),
        iteration_name='calm_alpaca_7_3',
        epochs=100,
        allowed_to_run=False
    ),
    ModelIteration(
        model_architecture=ModelArchitecture(
            architecture=tf.keras.Sequential(
                [
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
                    tf.keras.layers.Conv2D(128, 3, activation='relu'),
                    tf.keras.layers.MaxPooling2D(),
                    tf.keras.layers.Flatten(),
                    tf.keras.layers.Dense(4096, activation='relu'),
                    tf.keras.layers.BatchNormalization(),
                    tf.keras.layers.Dense(constants.NUM_CLASSES, activation='softmax')
                ]
            ),
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=["accuracy"]
        ),
        iteration_name='sweet_hedgehog_7_3_bn',
        epochs=100,
        allowed_to_run=False
    ),
    ModelIteration(
        model_architecture=ModelArchitecture(
            architecture=tf.keras.Sequential(
                [
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
                    tf.keras.layers.Conv2D(128, 3, activation='relu'),
                    tf.keras.layers.MaxPooling2D(),
                    tf.keras.layers.Conv2D(256, 3, activation='relu', padding="same"),
                    tf.keras.layers.Conv2D(256, 3, activation='relu', padding="same"),
                    tf.keras.layers.Conv2D(256, 3, activation='relu', padding="same"),
                    tf.keras.layers.Flatten(),
                    tf.keras.layers.Dense(4096, activation='relu'),
                    tf.keras.layers.BatchNormalization(),
                    tf.keras.layers.Dense(constants.NUM_CLASSES, activation='softmax')
                ]
            ),
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=["accuracy"]
        ),
        iteration_name='angry_dalmatian_7_3_bn',
        epochs=100,
        allowed_to_run=False
    ),
    ModelIteration(
        model_architecture=ModelArchitecture(
            architecture=tf.keras.Sequential(
                [
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
                    tf.keras.layers.Conv2D(128, 3, activation='relu'),
                    tf.keras.layers.MaxPooling2D(),
                    tf.keras.layers.Conv2D(256, 3, activation='relu', padding="same"),
                    tf.keras.layers.Conv2D(256, 3, activation='relu', padding="same"),
                    tf.keras.layers.Conv2D(256, 3, activation='relu', padding="same"),
                    tf.keras.layers.Flatten(),
                    tf.keras.layers.Dense(1024, activation='relu'),
                    tf.keras.layers.BatchNormalization(),
                    tf.keras.layers.Dense(64, activation='relu'),
                    tf.keras.layers.Dropout(0.2),
                    tf.keras.layers.Dense(constants.NUM_CLASSES, activation='softmax')
                ]
            ),
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=["accuracy"]
        ),
        iteration_name='friendly_elf_10_2_bn_do',
        epochs=100,
        allowed_to_run=False
    ),
    ModelIteration(
        model_architecture=ModelArchitecture(
            architecture=tf.keras.Sequential(
                [
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
                    tf.keras.layers.Conv2D(128, 3, activation='relu'),
                    tf.keras.layers.MaxPooling2D(),
                    tf.keras.layers.Conv2D(256, 3, activation='relu', padding="same"),
                    tf.keras.layers.Conv2D(256, 3, activation='relu', padding="same"),
                    tf.keras.layers.Conv2D(256, 3, activation='relu', padding="same"),
                    tf.keras.layers.Flatten(),
                    tf.keras.layers.Dense(1024, activation='relu'),
                    tf.keras.layers.BatchNormalization(),
                    tf.keras.layers.Dense(constants.NUM_CLASSES, activation='softmax')
                ]
            ),
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=["accuracy"]
        ),
        iteration_name='hungry_chicken_10_2_bn_do',
        epochs=100,
        allowed_to_run=False
    ),
    ModelIteration(
        model_architecture=ModelArchitecture(
            architecture=tf.keras.Sequential(
                [
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
                    tf.keras.layers.Dense(4096, activation='relu', kernel_regularizer='l2'),
                    tf.keras.layers.Dense(constants.NUM_CLASSES, activation='softmax')
                ]
            ),
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=["accuracy"]
        ),
        iteration_name='arrogant_dwarf_6_2_l2',
        epochs=100,
        allowed_to_run=False
    ),
    ModelIteration(
        model_architecture=ModelArchitecture(
            architecture=tf.keras.Sequential(
                [
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
                    tf.keras.layers.Dense(1024, activation='relu', kernel_regularizer='l2'),
                    tf.keras.layers.Dense(constants.NUM_CLASSES, activation='softmax')
                ]
            ),
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=["accuracy"]
        ),
        iteration_name='sporty_kangaroo_6_2_l2',
        epochs=100,
        allowed_to_run=False
    ),
    ModelIteration(
        model_architecture=ModelArchitecture(
            architecture=tf.keras.Sequential(
                [
                    tf.keras.layers.Conv2D(32, 3, activation='relu'),
                    tf.keras.layers.Conv2D(32, 3, activation='relu'),
                    tf.keras.layers.MaxPooling2D(),
                    tf.keras.layers.Conv2D(64, 3, activation='relu'),
                    tf.keras.layers.Conv2D(64, 3, activation='relu'),
                    tf.keras.layers.MaxPooling2D(),
                    tf.keras.layers.Conv2D(128, 3, activation='relu'),
                    tf.keras.layers.Conv2D(128, 3, activation='relu'),
                    tf.keras.layers.MaxPooling2D(),
                    tf.keras.layers.Conv2D(256, 3, activation='relu'),
                    tf.keras.layers.Conv2D(256, 3, activation='relu'),
                    tf.keras.layers.MaxPooling2D(),
                    tf.keras.layers.Conv2D(512, 3, activation='relu'),
                    tf.keras.layers.Conv2D(512, 3, activation='relu'),
                    tf.keras.layers.MaxPooling2D(),
                    tf.keras.layers.Flatten(),
                    tf.keras.layers.Dense(1024, activation='relu', kernel_regularizer='l2'),
                    tf.keras.layers.Dense(constants.NUM_CLASSES, activation='softmax')
                ]
            ),
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=["accuracy"]
        ),
        iteration_name='big_elefant_10_2_l2',
        epochs=100,
        allowed_to_run=False
    ),
    ModelIteration(
        model_architecture=ModelArchitecture(
            architecture=tf.keras.Sequential(
                [
                    tf.keras.layers.Conv2D(32, 3, activation='relu'),
                    tf.keras.layers.Conv2D(32, 3, activation='relu'),
                    tf.keras.layers.MaxPooling2D(),
                    tf.keras.layers.Conv2D(64, 3, activation='relu'),
                    tf.keras.layers.Conv2D(64, 3, activation='relu'),
                    tf.keras.layers.MaxPooling2D(),
                    tf.keras.layers.Conv2D(128, 3, activation='relu'),
                    tf.keras.layers.Conv2D(128, 3, activation='relu'),
                    tf.keras.layers.MaxPooling2D(),
                    tf.keras.layers.Conv2D(256, 3, activation='relu'),
                    tf.keras.layers.Conv2D(256, 3, activation='relu'),
                    tf.keras.layers.MaxPooling2D(),
                    tf.keras.layers.Conv2D(512, 3, activation='relu'),
                    tf.keras.layers.Conv2D(512, 3, activation='relu'),
                    tf.keras.layers.MaxPooling2D(),
                    tf.keras.layers.Flatten(),
                    tf.keras.layers.Dense(4096, activation='relu', kernel_regularizer='l2'),
                    tf.keras.layers.Dense(4096, activation='relu', kernel_regularizer='l2'),
                    tf.keras.layers.Dense(constants.NUM_CLASSES, activation='softmax')
                ]
            ),
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=["accuracy"]
        ),
        iteration_name='annoying_tortoise_10_3_l2',
        epochs=100,
        allowed_to_run=False
    ),
    ModelIteration(
        model_architecture=ModelArchitecture(
            architecture=tf.keras.Sequential(
                [
                    tf.keras.layers.Conv2D(32, 3, activation='relu'),
                    tf.keras.layers.BatchNormalization(),
                    tf.keras.layers.Conv2D(32, 3, activation='relu'),
                    tf.keras.layers.BatchNormalization(),
                    tf.keras.layers.MaxPooling2D(),
                    tf.keras.layers.Conv2D(64, 3, activation='relu'),
                    tf.keras.layers.BatchNormalization(),
                    tf.keras.layers.Conv2D(64, 3, activation='relu'),
                    tf.keras.layers.BatchNormalization(),
                    tf.keras.layers.MaxPooling2D(),
                    tf.keras.layers.BatchNormalization(),
                    tf.keras.layers.Conv2D(128, 3, activation='relu'),
                    tf.keras.layers.BatchNormalization(),
                    tf.keras.layers.Conv2D(128, 3, activation='relu'),
                    tf.keras.layers.BatchNormalization(),
                    tf.keras.layers.MaxPooling2D(),
                    tf.keras.layers.Conv2D(256, 3, activation='relu'),
                    tf.keras.layers.BatchNormalization(),
                    tf.keras.layers.Conv2D(256, 3, activation='relu'),
                    tf.keras.layers.BatchNormalization(),
                    tf.keras.layers.MaxPooling2D(),
                    tf.keras.layers.Conv2D(512, 3, activation='relu'),
                    tf.keras.layers.BatchNormalization(),
                    tf.keras.layers.Conv2D(512, 3, activation='relu'),
                    tf.keras.layers.BatchNormalization(),
                    tf.keras.layers.MaxPooling2D(),
                    tf.keras.layers.Flatten(),
                    tf.keras.layers.Dense(4096, activation='relu', kernel_regularizer='l2'),
                    tf.keras.layers.Dense(4096, activation='relu', kernel_regularizer='l2'),
                    tf.keras.layers.Dense(constants.NUM_CLASSES, activation='softmax')
                ]
            ),
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=["accuracy"]
        ),
        iteration_name='sleepy_tortoise_10_3',
        epochs=10,
        allowed_to_run=False
    ),
    ModelIteration(
        model_architecture=ModelArchitecture(
            architecture=tf.keras.Sequential(
                [
                    tf.keras.layers.Conv2D(32, 3, activation='relu'),
                    tf.keras.layers.BatchNormalization(),
                    tf.keras.layers.Conv2D(32, 3, activation='relu'),
                    tf.keras.layers.BatchNormalization(),
                    tf.keras.layers.MaxPooling2D(),
                    tf.keras.layers.Conv2D(64, 3, activation='relu'),
                    tf.keras.layers.BatchNormalization(),
                    tf.keras.layers.Conv2D(64, 3, activation='relu'),
                    tf.keras.layers.BatchNormalization(),
                    tf.keras.layers.MaxPooling2D(),
                    tf.keras.layers.BatchNormalization(),
                    tf.keras.layers.Conv2D(128, 3, activation='relu'),
                    tf.keras.layers.BatchNormalization(),
                    tf.keras.layers.Conv2D(128, 3, activation='relu'),
                    tf.keras.layers.BatchNormalization(),
                    tf.keras.layers.MaxPooling2D(),
                    tf.keras.layers.Conv2D(256, 3, activation='relu'),
                    tf.keras.layers.BatchNormalization(),
                    tf.keras.layers.Conv2D(256, 3, activation='relu'),
                    tf.keras.layers.BatchNormalization(),
                    tf.keras.layers.MaxPooling2D(),
                    tf.keras.layers.Conv2D(512, 3, activation='relu'),
                    tf.keras.layers.BatchNormalization(),
                    tf.keras.layers.Conv2D(512, 3, activation='relu'),
                    tf.keras.layers.BatchNormalization(),
                    tf.keras.layers.MaxPooling2D(),
                    tf.keras.layers.Flatten(),
                    tf.keras.layers.Dense(4096, activation='relu', kernel_regularizer='l1_l2'),
                    tf.keras.layers.Dense(4096, activation='relu', kernel_regularizer='l1_l2'),
                    tf.keras.layers.Dense(constants.NUM_CLASSES, activation='softmax')
                ]
            ),
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=["accuracy"]
        ),
        iteration_name='crazy_diamond_10_3',
        epochs=10,
        allowed_to_run=False
    ),
    ModelIteration(
        model_architecture=ModelArchitecture(
            architecture=tf.keras.Sequential(
                [
                    tf.keras.layers.Conv2D(32, 3, activation='relu'),
                    tf.keras.layers.BatchNormalization(),
                    tf.keras.layers.Conv2D(32, 3, activation='relu'),
                    tf.keras.layers.BatchNormalization(),
                    tf.keras.layers.MaxPooling2D(),
                    tf.keras.layers.Conv2D(64, 3, activation='relu'),
                    tf.keras.layers.BatchNormalization(),
                    tf.keras.layers.Conv2D(64, 3, activation='relu'),
                    tf.keras.layers.BatchNormalization(),
                    tf.keras.layers.MaxPooling2D(),
                    tf.keras.layers.BatchNormalization(),
                    tf.keras.layers.Conv2D(128, 3, activation='relu'),
                    tf.keras.layers.BatchNormalization(),
                    tf.keras.layers.Conv2D(128, 3, activation='relu'),
                    tf.keras.layers.BatchNormalization(),
                    tf.keras.layers.MaxPooling2D(),
                    tf.keras.layers.Conv2D(256, 3, activation='relu'),
                    tf.keras.layers.BatchNormalization(),
                    tf.keras.layers.Conv2D(256, 3, activation='relu'),
                    tf.keras.layers.BatchNormalization(),
                    tf.keras.layers.MaxPooling2D(),
                    tf.keras.layers.Conv2D(512, 3, activation='relu'),
                    tf.keras.layers.BatchNormalization(),
                    tf.keras.layers.Conv2D(512, 3, activation='relu'),
                    tf.keras.layers.BatchNormalization(),
                    tf.keras.layers.Conv2D(512, 3, activation='relu'),
                    tf.keras.layers.BatchNormalization(),
                    tf.keras.layers.Conv2D(512, 3, activation='relu'),
                    tf.keras.layers.BatchNormalization(),
                    tf.keras.layers.MaxPooling2D(),
                    tf.keras.layers.Flatten(),
                    tf.keras.layers.Dense(4096, activation='relu', kernel_regularizer='l2'),
                    tf.keras.layers.Dense(4096, activation='relu', kernel_regularizer='l2'),
                    tf.keras.layers.Dense(constants.NUM_CLASSES, activation='softmax')
                ]
            ),
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=["accuracy"]
        ),
        iteration_name='howling_wolf_14_3',
        epochs=10,
        allowed_to_run=True
    ),
    ModelIteration(
        model_architecture=ModelArchitecture(
            architecture=transfer_learning_model_1,
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
            loss=tf.keras.losses.CategoricalCrossentropy(),
            metrics=["accuracy"]
        ),
        iteration_name='transfer_learning_vgg16',
        epochs=100,
        allowed_to_run=True
    ),
    ModelIteration(
        model_architecture=ModelArchitecture(
            architecture=transfer_learning_model_2,
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
            loss=tf.keras.losses.CategoricalCrossentropy(),
            metrics=["accuracy"]
        ),
        iteration_name='transfer_learning_vgg19',
        epochs=100,
        allowed_to_run=True
    ),
    ModelIteration(
        model_architecture=ModelArchitecture(
            architecture=transfer_learning_model_3,
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
            loss=tf.keras.losses.CategoricalCrossentropy(),
            metrics=["accuracy"]
        ),
        iteration_name='transfer_learning_resnet101',
        epochs=100,
        allowed_to_run=True
    ),
    ModelIteration(
        model_architecture=ModelArchitecture(
            architecture=transfer_learning_model,
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
            loss=tf.keras.losses.CategoricalCrossentropy(),
            metrics=["accuracy"]
        ),
        iteration_name='transfer_learning_rn50_2',
        epochs=100,
        allowed_to_run=True
    )
]

if __name__ == '__main__':
    # Print the gpus available (test for local development)
    print("GPUs available: ", tf.config.list_physical_devices('GPU'))
    for model_iteration in model_iterations:
        if model_iteration.is_allowed_to_run():
            model_iteration.run()
