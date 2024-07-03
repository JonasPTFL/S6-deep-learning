import os
import sys

sys.path.append(os.getcwd())
import src.model_code.data_loader as data_loader


import tensorflow as tf
from src.model_code.model_persistence import model_load

INPUT_FILE = "student_1.keras"
OUTPUT_FILE = "student_1.tflite"

model = model_load(INPUT_FILE)

dataset = data_loader.DataLoader().load_data(subset="training")

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

tflite_model = converter.convert()

# Save the model.
with open(OUTPUT_FILE, 'wb') as f:
    f.write(tflite_model)


# check if tf lite model is correctly converted
with open(OUTPUT_FILE, 'rb') as f:
    tflite_model = f.read()

# Load TFLite model and allocate tensors to check if conversion was successful
interpreter = tf.lite.Interpreter(model_content=tflite_model)
interpreter.allocate_tensors()
print("Model converted successfully")

# output model summary
model.summary()
