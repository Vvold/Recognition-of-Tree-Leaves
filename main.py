from flask import Flask, render_template, request

from data_model import *


app = Flask (__name__)

@app.route('/', methods = ['GET'])
def hello():
    return render_template('index.html')


@app.route('/', methods = ['POST'])
def predict():
    imagefile = request.files['imagefile']
    image_path = "./images/" + imagefile.filename
    imagefile.save(image_path)
    try:
        image = img_preprocessing(image_path)
        predictions = model.predict(image)
        if predictions.max() > 0.9:
            return render_template('index.html', prediction="Sorry, maybe you didn't choose a picture with a leaf")
        else:
            predictions = np.argmax(predictions)
            classification = class_name[predictions]
            return render_template('index.html', prediction=f"Kind of tree:{classification}")
    except UnidentifiedImageError:
        return render_template('index.html', prediction='The file is not an image')


if __name__ == '__main__':
    app.run(port=3000, debug=True)











#integrity="sha384-BmbxuPwQa21c/FVzBcNJ7UAyJxM6wuqIj61tLrc4wSX0szH/Ev+nYRRuWlof1f1"