import numpy as np

from PIL import UnidentifiedImageError
from flask import Flask, render_template, request
from src.data_processing import Images
from src.train import class_names, path_to_model, SIZE
from keras.models import load_model

model = load_model(f'{path_to_model}')

app = Flask(__name__)


@app.route('/', methods=['GET'])
def hello():
    return render_template('index.html')


@app.route('/', methods=['POST'])
def predict():
    imagefile = request.files['imagefile']
    image_path = "./images/" + imagefile.filename
    try:
        imagefile.save(image_path)
        image, _ = Images(size=SIZE).image_to_array(path=image_path)
        predictions = model.predict(image)
        predictions = np.argmax(predictions)
        classification = class_names[predictions]
        return render_template('index.html', prediction=f'Kind of tree:{classification}')
    except UnidentifiedImageError:
        return render_template('index.html', prediction='The file is not an image')
    except PermissionError:
        return render_template('index.html', prediction='Please, select a file')


if __name__ == '__main__':
    app.run(port=8087, debug=True)
