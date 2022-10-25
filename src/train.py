import numpy as np
import yaml

from src.load_data import Images
from src.model import load_train_data, fit_model, save_model

#
config = yaml.safe_load(open("../config/params.yaml"))

path_load = config['path']['load']
path_write = config['path']['write']
class_names = config['images']['class_names']
SIZE = config['images']['SIZE']
path_to_model = config['model']['model']
test_size = config['model']['test_size']

if __name__ == "__main__":
    # test_work = Images()
    # X, y = test_work.data_img_pr eprocessing(path_load=path_load, class_name_dict=class_name_dict)
    # test_work.save_data(X, y, path_write)

    X, y = load_train_data(path_write)
    model = fit_model(size=SIZE, X=X, y=y, test_size=test_size)
    save_model(model=model, path_to_model=path_write)
