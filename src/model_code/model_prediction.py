import tensorflow as tf
import src.util.constants as constants
import src.util.default_values as default_values
import src.model_code.model_architecture as model_architecture
import src.util.default_values as default_values
from PIL import Image
import numpy as np
from skimage import transform

def model_predict(model: tf.keras.Model, image_path, model_img_height = default_values.img_height, model_img_width = default_values.img_width) -> str:
    image = load_image(image_path, model_img_height, model_img_width)
    return model.predict(image, model_img_height, model_img_width)


def load_image(image_path, model_img_height, model_img_width):
   np_image = Image.open(image_path)
   np_image = np.array(np_image).astype('float32')/255
   np_image = transform.resize(np_image, (model_img_height, model_img_width, 3))
   np_image = np.expand_dims(np_image, axis=0)
   return np_image
