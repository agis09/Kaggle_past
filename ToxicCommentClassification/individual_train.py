import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.layers import Input, Conv1D, Dense, MaxPool1D, Flatten, Dropout, GlobalMaxPool1D
from keras.models import Sequential
from keras.models import Model
from keras import backend as K
from keras.layers import Activation
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
import matplotlib.pyplot as plt
from keras.layers.normalization import BatchNormalization
import sys

df = pd.read_csv("train.csv", encoding="utf-8")


# sys.exit()

input_len = 300
input_len2 = 1000000


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
            tmp = np.append(tmp, np.array([make_one_hot(char) for char in word]))
            cnt -= 1
            if cnt == 0:
                break
        if cnt > 0:
            for i in range (0, cnt):
                tmp = np.append(tmp, np.zeros(27))
        tmp = np.reshape(tmp, (input_len, 27))
        lst = np.append(lst, np.array(tmp))
    lst = np.reshape(lst, (len(x), input_len, 27))
    return lst


def batch_iter(data_set, label, batch_size, shuffle=True):
    num_batches_per_epoch = int((len(data_set)-1)/batch_size)+1

    def data_generator():
        data_size = len(data_set)
        while True:
            if shuffle:
                shuffle_indices = np.random.permutation(np.arange(data_size))
                shuffled_data = data_set[shuffle_indices]
                shuffled_labels = label[shuffle_indices]
            else:
                shuffled_data = data_set
                shuffled_labels = label

            for batch_num in range(num_batches_per_epoch):
                start_index = batch_num*batch_size
                end_index = min((batch_num+1)*batch_size, data_size)
                x = shuffled_data[start_index:end_index]
                y = shuffled_labels[start_index:end_index]
                yield make_input(x), y
    return num_batches_per_epoch, data_generator()


def build_model(kernel_sizes, dense_units,
                vocab_size, nb_filter, nb_class, keep_prob, maxlen):
    inputs = Input(batch_shape=(None, maxlen, vocab_size))
    drop = Dropout(0.2)(inputs)
    conv1 = Conv1D(nb_filter, kernel_sizes[0], activation='relu')(drop)
    # norm1 = BatchNormalization()(conv1)
    pool1 = MaxPool1D(pool_size=3)(conv1)
    drop1 = Dropout(0)(pool1)
    conv2 = Conv1D(nb_filter, kernel_sizes[1], activation='relu')(drop1)
    # norm2 = BatchNormalization()(conv2)
    pool2 = MaxPool1D(pool_size=3)(conv2)
    drop2 = Dropout(0)(pool2)

    # conv3 = Conv1D(nb_filter, kernel_sizes[2], activation='relu')(drop2)
    # drop3 = Dropout(0.1)(conv3)
    # pool3 = MaxPool1D(pool_size=3)(conv3)
    # norm3 = BatchNormalization()(drop3)
    # conv4 = Conv1D(nb_filter, kernel_sizes[3], activation='relu')(norm3)
    # drop4 = Dropout(0.1)(conv4)
    # pool4 = MaxPool1D(pool_size=3)(conv4)
    # norm4 = BatchNormalization()(conv4)
    # conv5 = Conv1D(nb_filter, kernel_sizes[4], activation='relu')(drop4)
    # drop5 = Dropout(0.1)(conv5)
    # pool5 = MaxPool1D(pool_size=3)(conv5)
    # norm5 = BatchNormalization()(conv5)
    # conv6 = Conv1D(nb_filter, kernel_sizes[5], activation='relu')(drop5)
    # norm6 = BatchNormalization()(conv6)
    # pool6 = MaxPool1D(pool_size=3)(drop2)
    # drop3 = Dropout(0.25)(pool3)
    pool = Flatten()(drop2)
    # pool = GlobalMaxPool1D()(pool2)

    fc1 = Dense(dense_units[0], activation='relu')(pool)
    fc1 = Dropout(keep_prob)(fc1)
    fc2 = Dense(dense_units[1], activation='relu')(fc1)
    fc2 = Dropout(keep_prob)(fc2)
    pred = Dense(nb_class, activation='sigmoid')(fc2)

    model = Model(inputs=[inputs], outputs=[pred])

    return model


n_in = input_len
n_out = 1
kernel_size = [7, 7, 3, 3, 3, 3]
nb_filter =256
dense_units =[1024, 1024]
keep_prob = 0.5

CLASSES_LIST = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

for i in range(6):

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

    if i==0: model.summary()

    batch_size = 32
    epochs = 7

    np.random.permutation(df.index)

    df2 = df[(df["comment_text"].str.len() <= input_len2) & (df[CLASSES_LIST[i]] > 0)]

    data = df2.loc[:, "comment_text"].as_matrix().astype('str')
    labels_toxic = df2.loc[:, CLASSES_LIST[i]].as_matrix().astype('float16')

    df0 = df[(df["comment_text"].str.len() <= input_len2) & (df[CLASSES_LIST[i]] == 0)].reset_index(drop=True)

    nois_len = len(data)

    data0 = df0.loc[:nois_len, "comment_text"].as_matrix().astype('str')
    labels0_toxic = df0.loc[:nois_len, CLASSES_LIST[i]].as_matrix().astype('float16')

    data = np.append(data, data0)
    labels_toxic = np.append(labels_toxic, labels0_toxic)

    print(CLASSES_LIST[i])

    data_train, data_test, labels_train, labels_test = train_test_split(data, labels_toxic, train_size=0.8)

    train_steps, train_batches = batch_iter(data_train, labels_train, batch_size)
    valid_steps, valid_batches = batch_iter(data_test, labels_test, batch_size)

    model.fit_generator(train_batches, train_steps,
                        epochs=epochs,
                        validation_data=valid_batches,
                        validation_steps=valid_steps,
                        verbose=1)

    # /////////////////////save//////////////////////

    model.save_weights('w_' + CLASSES_LIST[i] + '.hdf5')
    json_string = model.to_json()
    open('model_' + CLASSES_LIST[i] + '.json', 'w').write(json_string)
