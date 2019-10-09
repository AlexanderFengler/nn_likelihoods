from tensorflow import keras
import numpy as np
from cdwiener import array_fptd
import os
import pandas as pd
import re
import time
from datetime import datetime
import pickle
import yaml

data_folder = "/users/afengler/data/kde/full_ddm/train_test_data_20000/"
files = os.listdir(data_folder)
files = [f for f in files if re.match("data_.*", f)]

# kde_make_train_test_split(folder = data_folder,
#                           p_train = 0.8)

# ix = np.arange(X.shape[0])
# np.random.shuffle(ix)
# X = X[ix]
# y = y[ix]

timestamp = datetime.now().strftime('%m_%d_%y_%H_%M_%S')
# save model configuration and hyperparams in folder

model_path  = "/users/afengler/data/tony/kde/full_ddm/keras_models"
model_path += "/" + "deep_inference" + timestamp

if not os.path.exists(model_path):
    os.makedirs(model_path)

def heteroscedastic_loss(true, pred):
    params = pred.shape[1] // 2
    point = pred[:, :params]
    var = pred[:, params:]
    precision = 1 / var
    return keras.backend.sum(precision * (true - point) ** 2 + keras.backend.log(var), -1)

inp = keras.Input(shape=(None, 2))
x = inp
x = keras.layers.Conv1D(64, kernel_size=1, strides=1, activation="relu")(x)
x = keras.layers.Conv1D(64, kernel_size=3, strides=2, activation="relu")(x)
x = keras.layers.Conv1D(128, kernel_size=3, strides=2, activation="relu")(x)
x = keras.layers.Conv1D(128, kernel_size=3, strides=2, activation="relu")(x)
x = keras.layers.Conv1D(128, kernel_size=3, strides=2, activation="relu")(x)
x = keras.layers.GlobalAveragePooling1D()(x)
mean = keras.layers.Dense(5)(x)
var = keras.layers.Dense(5, activation="softplus")(x)
out = keras.layers.Concatenate()([mean, var])

model = keras.Model(inp, out)

filename = model_path + "/model.h5"
earlystopper = keras.callbacks.EarlyStopping(monitor="val_loss", patience=2, verbose=1)
checkpoint = keras.callbacks.ModelCheckpoint(filename, monitor="val_loss", verbose=1, save_best_only=False)

# fit model

model.compile(loss=heteroscedastic_loss, optimizer="adam")
history = model.fit(X, y, validation_split=.01, epochs=10, batch_size=32, callbacks=[checkpoint, earlystopper])

print(history)

model.save(model_path + "/model_final.h5")

