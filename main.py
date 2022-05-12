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

    image = img_preprocessing(image_path)
    predictions = model.predict(image)
    predictions = np.argmax(predictions)

    classification = class_name[predictions]

    return render_template('index.html', prediction = classification)

if __name__ == '__main__':
    app.run(port=3000, debug=True)











#integrity="sha384-BmbxuPwQa21c/FVzBcNJ7UAyJxM6wuqIj61tLrc4wSX0szH/Ev+nYRRuWlof1f1"