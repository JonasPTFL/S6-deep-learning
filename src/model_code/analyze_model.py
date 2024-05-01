import tensorflow as tf
import src.util.constants as constants

# Load the saved model
loaded_model = tf.keras.models.load_model(constants.MODEL_PATH)

# Display the model architecture
loaded_model.summary()
