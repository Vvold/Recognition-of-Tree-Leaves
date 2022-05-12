# Recognition of Tree Leaves
____

##### Project Tasks:
Classification of photo leaves on what kind of tree the leaf belongs to using neural networks.
Created an app on Flask, which allows you to interact with the neural network through a nice interface on the browser tab

### Data
Data for neural network learning consists of pre-downloaded photos of tree leaves. Classification will be based on the
following types of tree leaves:
- *apple*
- *maple*
- *willow*
- *birch*
- *oak*


### Preprocessing
The neural network can not directly interact with the photo, for the correct operation of the network requires data processing:
- bring the size of the image to one view (224 * 244 pixels)
- translate color photo to `color_mode='grayscale'`
- assign a numerical value to each shade of gray
- scale data
- transfer data from one photo to one `np.array`

For data that will come to `prediction` will be the same pre-processing
## Classification
A recurrent neural network with one input, one hidden, and one output layer was used for classification.
``` python
model = Sequential()

model.add(Dense(800, input_dim = 50176, activation = "relu"))
model.add(Dropout(0.2))
model.add(Dense(5, activation = "softmax"))
#compile model
model.compile(loss = "categorical_crossentropy",
optimizer = "SGD", metrics = ["accuracy"])

```

### Flask app
Main page:

![main page](C:\my\Python\leaf_ML_add\result_of_work_img\2022-03-05 223630.png)

Main page after uploading a photo:

![download photo](C:\my\Python\leaf_ML_add\result_of_work_img\2022-03-05 224016.png)

Main page with the output of the classification result:

![prediction](C:\my\Python\leaf_ML_add\result_of_work_img\2022-03-05 224039.png)

## Contacts

---
### Vitkovskiy Volodymyr
- email: VitkovskyiVB@gmail.com
- telegram: @wvld_11
- github: [Vvold](https://github.com/Vvold)
