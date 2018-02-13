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

input_len = 1014

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



def build_model(kernel_sizes, dense_units,
                vocab_size, nb_filter, nb_class, keep_prob, maxlen):
    inputs = Input(batch_shape=(None, maxlen, vocab_size))

    conv1 = Conv1D(nb_filter, kernel_sizes[0], activation='relu')(inputs)
    pool1 = MaxPool1D(pool_size=3)(conv1)
    norm1 = BatchNormalization()(pool1)
    conv2 = Conv1D(nb_filter, kernel_sizes[1], activation='relu')(norm1)
    pool2 = MaxPool1D(pool_size=3)(conv2)
    norm2 = BatchNormalization()(pool2)

    conv3 = Conv1D(nb_filter, kernel_sizes[2], activation='relu')(norm2)
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



n_in = input_len
n_out = 1
kernel_size = [7, 7, 3, 3, 3, 3]
nb_filter = 256
dense_units =[1024, 1024]
keep_prob = 0.5


model = build_model(kernel_sizes=kernel_size,
                    dense_units=dense_units,
                    vocab_size=27,
                    nb_filter=nb_filter,
                    nb_class=1,
                    keep_prob=keep_prob,
                    maxlen=input_len)

model.compile(loss='binary_crossentropy',
              optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999),
              metrics=['accuracy'])


batch_size = 3
steps, generator = batch_iter(data, batch_size)


model.load_weights('w_toxic.hdf5')
hist_toxic = model.predict_generator(generator, steps,verbose=1)
model.load_weights('w_sev_toxic.hdf5')
hist_sev_toxic = model.predict_generator(generator, steps,verbose=1)
model.load_weights('w_obscene.hdf5')
hist_obscene = model.predict_generator(generator, steps,verbose=1)
model.load_weights('w_threat.hdf5')
hist_threat = model.predict_generator(generator, steps,verbose=1)
model.load_weights('w_insult.hdf5')
hist_insult = model.predict_generator(generator, steps,verbose=1)
model.load_weights('w_id_hate.hdf5')
hist_id_hate = model.predict_generator(generator, steps, verbose=1)

# /////////////////////save//////////////////////

with open('predict.csv', 'a', newline='', encoding="utf-8")as file:
    csvWriter = csv.writer(file)
    for i, k1,k2,k3,k4,k5,k6 in zip(id, hist_toxic,hist_sev_toxic,hist_obscene,hist_threat,hist_insult,hist_id_hate):
        sum = k1+k2+k3+k4+k5+k6
        ans = [i, k1/sum, k2/sum, k3/sum, k4/sum, k5/sum, k6/sum]
        csvWriter.writerow(ans)
    file.close()
