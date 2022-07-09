from keras.models import load_model
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
import pickle
from PIL import Image, UnidentifiedImageError
import numpy as np

SIZE = 224

# read from file class_name
with open('recognition model/class_name_dict', 'rb') as f:
    class_name = pickle.load(f)

model = load_model('recognition model/model.h5')


def img_preprocessing (image_path):
    # transform the image into shades of gray and reduce the size = SIZE*SIZE
    img = load_img(image_path, target_size=(SIZE, SIZE), color_mode='grayscale')
    #convert the image into an array
    img_arr = img_to_array(img)
    #convert the shape of the array to a flat vector
    img_arr = img_arr.reshape(1, SIZE*SIZE)
    #invert the image
    img_arr = 255 - img_arr
    #normalize the image
    img_arr /= 255
    return img_arr

