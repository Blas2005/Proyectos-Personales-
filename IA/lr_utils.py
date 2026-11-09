import numpy as np
import h5py

def load_dataset():
    train_dataset = h5py.File('datasets/train_catvnoncat.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # características de entrenamiento
    train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # etiquetas de entrenamiento

    test_dataset = h5py.File('datasets/test_catvnoncat.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:])   # características de prueba
    test_set_y_orig = np.array(test_dataset["test_set_y"][:])   # etiquetas de prueba

    classes = np.array(test_dataset["list_classes"][:])          # lista de clases

    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes