import tensorflow as tf
import src.util.constants as constants

# data loader
batch_size = 32
img_height = 224
img_width = 224
seed = 123
validation_split = 0.2

# model architecture
model_architecture = tf.keras.Sequential([
    tf.keras.layers.Rescaling(1. / 255),
    tf.keras.layers.Conv2D(32, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(32, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(32, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(constants.NUM_CLASSES, activation='softmax')
])
compile_metrics = ['accuracy']
loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)

# training
epochs = 10

# persistence
model_save_filename = 'model_v1.keras'
checkpoint_model_save_filename = 'model_v1.weights.h5'

# general
model_iteration_name = 'unnamed_iteration'
