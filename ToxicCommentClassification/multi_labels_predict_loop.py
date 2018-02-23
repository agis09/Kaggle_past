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
from keras.models import model_from_json
import matplotlib.pyplot as plt
from keras.layers.normalization import BatchNormalization
import csv

df = pd.read_csv("test.csv", encoding="utf-8")

data = df.loc[:, "comment_text"]
data = data.as_matrix().astype('str')
id = df.loc[:, "id"]

with open('predict.csv', 'a', newline='', encoding="utf-8")as file:
    csvWriter = csv.writer(file)
    csvWriter.writerow(["id", "toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"])
    file.close()

df = pd.read_csv("predict.csv", encoding="utf-8")
df['id']=id
df.to_csv("predict.csv", index=False)


input_len = 300

def make_one_hot(char):
    index = ord(char) - ord("a")
    value = np.zeros(27)
    if index >= 0 and index <= 25:
        value[index] = 1.0
    else:
        value[26] = 1.0
    return value


def make_input(x):
    lst = np.array([])
    for words in x:
        words = words.lower()
        tmp = np.array([])
        cnt = input_len
        for word in words:
            tmp = np.append(tmp, [make_one_hot(char) for char in word])
            cnt -= 1
            if cnt == 0:
                break
        if cnt > 0:
            for i in range (0, cnt):
                tmp = np.append(tmp, np.zeros(27))
        tmp = np.reshape(tmp, (input_len, 27))
        lst = np.append(lst, tmp)
    lst = np.reshape(lst, (len(x), input_len, 27))
    return lst


def batch_iter(data_set, batch_size, shuffle=False):
    num_batches_per_epoch = int((len(data_set)-1)/batch_size)+1

    def data_generator():
        data_size = len(data_set)
        while True:
            if shuffle:
                shuffle_indices = np.random.permutation(np.arange(data_size))
                shuffled_data = data_set[shuffle_indices]
            else:
                shuffled_data = data_set

            for batch_num in range(num_batches_per_epoch):
                start_index = batch_num*batch_size
                end_index = min((batch_num+1)*batch_size, data_size)
                x = shuffled_data[start_index:end_index]
                yield make_input(x)
    return num_batches_per_epoch, data_generator()

lr = 0.0001

for num in range(6):
    if num == 0:
        model = model_from_json(open('model_toxic.json').read())

        model.compile(loss='binary_crossentropy',
                      optimizer=Adam(lr=lr, beta_1=0.9, beta_2=0.999),
                      metrics=['accuracy'])

        batch_size = 15
        steps, generator = batch_iter(data, batch_size)

        hist = []

        model.load_weights('w_toxic.hdf5')

        hist_toxic = model.predict_generator(generator, steps, verbose=1)
        for i in hist_toxic:
            hist.extend(i)

        df2 = pd.read_csv("predict.csv", encoding="utf-8")
        df2['toxic'] = hist
        df2.to_csv("predict.csv", index=False)

    if num == 1:
        model = model_from_json(open('model_sev_toxic.json').read())

        model.compile(loss='binary_crossentropy',
                      optimizer=Adam(lr=lr, beta_1=0.9, beta_2=0.999),
                      metrics=['accuracy'])

        batch_size = 15
        steps, generator = batch_iter(data, batch_size)

        hist = []

        model.load_weights('w_sev_toxic.hdf5')

        hist_toxic = model.predict_generator(generator, steps, verbose=1)
        for i in hist_toxic:
            hist.extend(i)

        df2 = pd.read_csv("predict.csv", encoding="utf-8")
        df2['severe_toxic'] = hist
        df2.to_csv("predict.csv", index=False)

    if num == 2:
        model = model_from_json(open('model_obscene.json').read())

        model.compile(loss='binary_crossentropy',
                      optimizer=Adam(lr=lr, beta_1=0.9, beta_2=0.999),
                      metrics=['accuracy'])

        batch_size = 15
        steps, generator = batch_iter(data, batch_size)

        hist = []

        model.load_weights('w_obscene.hdf5')

        hist_toxic = model.predict_generator(generator, steps, verbose=1)
        for i in hist_toxic:
            hist.extend(i)

        df2 = pd.read_csv("predict.csv", encoding="utf-8")
        df2['obscene'] = hist
        df2.to_csv("predict.csv", index=False)

    if num == 3:
        model = model_from_json(open('model_threat.json').read())

        model.compile(loss='binary_crossentropy',
                      optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999),
                      metrics=['accuracy'])

        batch_size = 15
        steps, generator = batch_iter(data, batch_size)

        hist = []

        model.load_weights('w_threat.hdf5')

        hist_toxic = model.predict_generator(generator, steps, verbose=1)
        for i in hist_toxic:
            hist.extend(i)

        df2 = pd.read_csv("predict.csv", encoding="utf-8")
        df2['threat'] = hist
        df2.to_csv("predict.csv", index=False)

    if num == 4:
        model = model_from_json(open('model_insult.json').read())

        model.compile(loss='binary_crossentropy',
                      optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999),
                      metrics=['accuracy'])

        batch_size = 15
        steps, generator = batch_iter(data, batch_size)

        hist = []

        model.load_weights('w_insult.hdf5')

        hist_toxic = model.predict_generator(generator, steps, verbose=1)
        for i in hist_toxic:
            hist.extend(i)

        df2 = pd.read_csv("predict.csv", encoding="utf-8")
        df2['insult'] = hist
        df2.to_csv("predict.csv", index=False)

    if num == 5:
        model = model_from_json(open('model_id_hate.json').read())

        model.compile(loss='binary_crossentropy',
                      optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999),
                      metrics=['accuracy'])

        batch_size = 15
        steps, generator = batch_iter(data, batch_size)

        hist = []

        model.load_weights('w_id_hate.hdf5')

        hist_toxic = model.predict_generator(generator, steps, verbose=1)
        for i in hist_toxic:
            hist.extend(i)

        df2 = pd.read_csv("predict.csv", encoding="utf-8")
        df2['identity_hate'] = hist
        df2.to_csv("predict.csv", index=False)

    # /////////////////////save//////////////////////

