import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.layers import Input, Conv1D, Dense, MaxPool1D, Flatten, Dropout
from keras.models import Sequential
from keras.models import Model
from keras import backend as K
from keras.layers import Activation
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import csv

df = pd.read_csv("test.csv", encoding="utf-8")

x = df.loc[:, "comment_text"]
id = df.loc[:, "id"]


def make_one_hot(char):
    index = ord(char) - ord("a")
    value = np.zeros(26)
    if index >= 0 and index <= 25:
        value[index] = 1.0
    return value


data = np.array([])
for words in x:
    words = words.lower()
    tmp = np.array([])
    cnt = 1014
    for word in words:
        tmp = np.append(tmp, np.array([make_one_hot(char) for char in word]))
        cnt -= 1
        if cnt == 0:
            break
    if cnt > 0:
        for i in range(0, cnt):
            tmp = np.append(tmp, np.zeros(26))
    tmp = np.reshape(tmp, (1014, 26))
    data = np.append(data, tmp)
data = np.reshape(data, (len(x), 1014, 26))

np.save('test_data~', data)
data = np.load('test_data.npy')


def weight_variable(shape):
    return K.truncated_normal(shape, stddev=0.01)


def build_model(kernel_sizes, dense_units,
                vocab_size, nb_filter, nb_class, keep_prob, maxlen):
    inputs = Input(batch_shape=(None, maxlen, vocab_size))

    conv1 = Conv1D(nb_filter, kernel_sizes[0], activation='relu')(inputs)
    pool1 = MaxPool1D(pool_size=3)(conv1)

    conv2 = Conv1D(nb_filter, kernel_sizes[1], activation='relu')(pool1)
    pool2 = MaxPool1D(pool_size=3)(conv2)

    conv3 = Conv1D(nb_filter, kernel_sizes[2], activation='relu')(pool2)
    conv4 = Conv1D(nb_filter, kernel_sizes[3], activation='relu')(conv3)
    conv5 = Conv1D(nb_filter, kernel_sizes[4], activation='relu')(conv4)
    conv6 = Conv1D(nb_filter, kernel_sizes[5], activation='relu')(conv5)
    pool3 = MaxPool1D(pool_size=3)(conv6)
    pool3 = Flatten()(pool3)

    fc1 = Dense(dense_units[0], activation='relu')(pool3)
    fc1 = Dropout(keep_prob)(fc1)
    fc2 = Dense(dense_units[1], activation='relu')(fc1)
    fc2 = Dropout(keep_prob)(fc2)
    pred = Dense(nb_class, activation='softmax')(fc2)

    model = Model(inputs=[inputs], outputs=[pred])

    return model


n_in = len(data[0])
n_out = 6
kernel_size = [7, 7, 3, 3, 3, 3]
nb_filter = 256
dense_units =[1024, 1024]
keep_prob = 0.5


model = build_model(kernel_sizes=kernel_size,
                    dense_units=dense_units,
                    vocab_size=26,
                    nb_filter=nb_filter,
                    nb_class=6,
                    keep_prob=keep_prob,
                    maxlen=1014)

model.compile(loss='categorical_crossentropy',
              optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999),
              metrics=['accuracy'])

model.load_weights('test.hdf5')
hist = model.predict(data)


# /////////////////////save//////////////////////
with open('predict.csv', 'a', newline='')as file:
    csvWriter = csv.writer(file)
    csvWriter.writerow(["id", "comment_text", "toxic","savere_toxic","obscene", "threat", "insult", "identity_hate" ])
    for i, j, k in zip(id, x, hist):
        csvWriter.writerow([i, j, k])