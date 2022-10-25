import os
import pickle
import PIL.Image
import matplotlib.pyplot as plt

from keras.utils import load_img, img_to_array
from IPython.display import Image


class Images:

    def __init__(self, size=100):
        self.size = size

    def img_preprocessing(self, image_path: str) -> tuple[list, PIL.Image.Image]:
        # transform the image
        img = load_img(image_path, target_size=(self.size, self.size), color_mode='grayscale')
        # convert the image into an array
        img_arr = img_to_array(img)
        img_arr = img_arr.reshape(1, self.size * self.size)
        img_arr = 255 - img_arr
        img_arr /= 255
        img_arr = list(img_arr[0])
        return img_arr, img

    def data_img_preprocessing(self, path_load: str, class_name_dict: dict, limit_load=10) -> tuple[list, list]:
        train_img_array = []
        train_img_target = []

        for files in class_name_dict.keys():
            photo = os.listdir(f'../{path_load}/{class_name_dict.setdefault(files)}')
            if len(photo) >= limit_load:
                for leaf in photo:
                    image, _ = self.img_preprocessing(f'../{path_load}/{class_name_dict.setdefault(files)}/{leaf}')
                    train_img_array.append(image)
                    train_img_target.append(files)
        return train_img_array, train_img_target

    def show_img(self, path: str) -> None:
        Image(path)
        _, processed_image = self.img_preprocessing(path)
        plt.imshow(processed_image, cmap=plt.cm.binary)
        plt.show()

    def save_data(self, X: list, y: list, path_write: str) -> None:
        # save X, y
        with open(f'../{path_write}/X', 'wb') as fp:
            pickle.dump(X, fp)
        with open(f'../{path_write}/y', 'wb') as fp:
            pickle.dump(y, fp)


if __name__ == "__main__":
    test_img_show = Images()
    test_img_show.show_img("../recognition model/test1.jpg")
