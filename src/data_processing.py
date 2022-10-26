import os
import pickle
import PIL.Image
import matplotlib.pyplot as plt

from keras.utils import load_img, img_to_array


class Images:

    def __init__(self, size=100):
        self.size = size

    def image_to_array(self, path: str) -> tuple[list, PIL.Image.Image]:
        # transform the image
        img = load_img(path, target_size=(self.size, self.size), color_mode='grayscale')
        # image to array
        img_arr = img_to_array(img)
        img_arr = img_arr.reshape(1, self.size * self.size)
        img_arr = 255 - img_arr
        img_arr /= 255
        img_arr = list(img_arr[0])
        return img_arr, img

    def show_image(self, path: str) -> None:
        _, processed_image = self.image_to_array(path)
        plt.imshow(processed_image)
        plt.show()

    def data_preparation(self, path: str, class_names: dict, limit_load=10) -> tuple[list, list]:
        images = []
        targets = []

        for files in class_names.keys():
            photo = os.listdir(f'../{path}/{class_names.setdefault(files)}')
            if len(photo) >= limit_load:
                for leaf in photo:
                    image, _ = self.image_to_array(f'../{path}/{class_names.setdefault(files)}/{leaf}')
                    images.append(image)
                    targets.append(files)
        return images, targets

    def save_data(self, images: list, targets: list, path: str) -> None:
        # save X, y
        with open(f'../{path}/images', 'wb') as fp:
            pickle.dump(images, fp)
        with open(f'../{path}/targets', 'wb') as fp:
            pickle.dump(targets, fp)


if __name__ == "__main__":
    test_img_show = Images()
    test_img_show.show_image("../recognition model/test1.jpg")
