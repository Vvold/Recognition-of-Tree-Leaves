import pickle

import keras
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score


def load_train_data(path_to_data: str) -> tuple[np.array, np.array]:
    # open X, y
    with open(f'../{path_to_data}/X', 'rb') as fp:
        X = pickle.load(fp)
    with open(f'../{path_to_data}/y', 'rb') as fp:
        y = pickle.load(fp)

    # Convert list X and y the sample into np.array
    X = np.array(X)
    y = np.array(y)
    y = to_categorical(y, 5)
    return X, y


def fit_model(X: np.array, y: np.array, size: int, test_size: float) -> keras.Sequential:
    # split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y)
    # Initializing the Recurrent Neural Network
    model = Sequential()
    model.add(Dense(800, input_dim=size * size, activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(5, activation="softmax"))
    # compile model
    model.compile(loss="categorical_crossentropy", optimizer="SGD", metrics=["accuracy"])
    # fit model
    model.fit(X_train, y_train, epochs=40, verbose=False)
    # evaluate the model
    y_test = np.argmax(y_test, axis=1)
    y_pred = np.argmax(model.predict(X_test), axis=1)
    f1 = f1_score(y_test, y_pred, average="micro")
    print(f'F1 score = {f1}')
    return model


def save_model(model, path_to_model: str) -> None:
    model.save(f'../{path_to_model}/model.h5')


if __name__ == "__main__":
    pass
