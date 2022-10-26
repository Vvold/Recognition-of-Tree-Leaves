import yaml

from src.data_processing import Images
from src.model import load_data, fit_model, save_model

config = yaml.safe_load(open("../config/params.yaml"))
data_folder = config['path']['folder']
load_path = config['path']['load']
class_names = config['image']['class_names']
SIZE = config['image']['SIZE']
limit_load = config['image']['limit_load']
path_to_model = config['model']['path_model']
test_size = config['model']['test_size']
epochs = config['model']['epochs']

if __name__ == "__main__":
    data = Images(SIZE)
    images, targets = data.data_preparation(path=data_folder, class_names=class_names, limit_load=limit_load)
    data.save_data(path=load_path, images=images, targets=targets)

    X, y = load_data(path=load_path)
    model = fit_model(X=X, y=y, size=SIZE, test_size=test_size, epochs=epochs)
    save_model(model=model, path=load_path)
