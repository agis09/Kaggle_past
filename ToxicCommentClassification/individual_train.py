import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.layers import Input, Conv1D, Dense, MaxPool1D, Flatten, Dropout, Embedding
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

    conv1 = Conv1D(nb_filter, kernel_sizes[0], activation='relu')(inputs)
    norm1 = BatchNormalization()(conv1)
    pool1 = MaxPool1D(pool_size=3)(norm1)
    conv2 = Conv1D(nb_filter, kernel_sizes[1], activation='relu')(pool1)
    norm2 = BatchNormalization()(conv2)
    # pool2 = MaxPool1D(pool_size=3)(norm2)

    # conv3 = Conv1D(nb_filter, kernel_sizes[2], activation='relu')(pool2)
    # norm3 = BatchNormalization()(conv3)
    # conv4 = Conv1D(nb_filter, kernel_sizes[3], activation='relu')(conv3)
    # norm4 = BatchNormalization()(conv4)
    # conv5 = Conv1D(nb_filter, kernel_sizes[4], activation='relu')(conv4)
    # norm5 = BatchNormalization()(conv5)
    # conv6 = Conv1D(nb_filter, kernel_sizes[5], activation='relu')(conv5)
    # norm6 = BatchNormalization()(conv6)
    pool3 = MaxPool1D(pool_size=3)(conv2)
    pool3 = Flatten()(pool3)

    fc1 = Dense(dense_units[0])(norm2)
    # norm7 = BatchNormalization()(fc1)
    fc_a = Activation('relu')(fc1)
    fc_b = Dropout(keep_prob)(fc_a)
    fc2 = Dense(dense_units[1])(fc_b)
    # norm8 = BatchNormalization()(fc2)
    fc_c = Activation('relu')(fc2)
    fc_d = Dropout(keep_prob)(fc_c)
    pred = Dense(nb_class, activation='sigmoid')(fc_d)

    model = Model(inputs=[inputs], outputs=[pred])

    return model


n_in = input_len
n_out = 1
kernel_size = [7, 7, 3, 3, 3, 3]
nb_filter = 256
dense_units =[1024, 1024]
keep_prob = 0.5


lr = 0.001

for i in range(6):

    model = build_model(kernel_sizes=kernel_size,
                        dense_units=dense_units,
                        vocab_size=27,
                        nb_filter=nb_filter,
                        nb_class=1,
                        keep_prob=keep_prob,
                        maxlen=input_len)

    model.compile(loss='binary_crossentropy',
                  optimizer=Adam(lr=lr, beta_1=0.9, beta_2=0.999),
                  metrics=['accuracy'])
    model.summary()

    batch_size = 50
    epochs = 10

    if i==0 :
        ######################################toxic############################################

        df2 = df[(df["comment_text"].str.len() <=  input_len2) & (df["toxic"] > 0)]

        data = df2.loc[:, "comment_text"].as_matrix().astype('str')
        labels_toxic = df2.loc[:, "toxic"].as_matrix().astype('float16')

        df0 = df[(df["comment_text"].str.len() <=  input_len2) & (df["toxic"] == 0)]

        nois_len = len(data)

        data0 = df0.loc[:nois_len, "comment_text"].as_matrix().astype('str')
        labels0_toxic = df0.loc[:nois_len, "toxic"].as_matrix().astype('float16')

        data = np.append(data, data0)
        labels_toxic = np.append(labels_toxic, labels0_toxic)

        print("toxic")
        data_train, data_test, labels_train, labels_test = train_test_split(data, labels_toxic, train_size=0.8)

        train_steps, train_batches = batch_iter(data_train, labels_train, batch_size)
        valid_steps, valid_batches = batch_iter(data_test, labels_test, batch_size)

        model.fit_generator(train_batches, train_steps,
                            epochs=epochs,
                            validation_data=valid_batches,
                            validation_steps=valid_steps)

        # /////////////////////save//////////////////////

        model.save_weights('w_toxic.hdf5')
        json_string = model.to_json()
        open('model_toxic.json', 'w').write(json_string)
    elif i==1:
        ######################################sev_toxic############################################

        df2 = df[(df["comment_text"].str.len() <= input_len2) & (df["severe_toxic"] > 0)]

        data = df2.loc[:, "comment_text"].as_matrix().astype('str')
        labels_sev_toxic = df2.loc[:, "severe_toxic"].as_matrix().astype('float16')

        df0 = df[(df["comment_text"].str.len() <= input_len2) & (df["severe_toxic"] == 0)]

        nois_len = len(data)

        data0 = df0.loc[:nois_len, "comment_text"].as_matrix().astype('str')
        labels0_sev_toxic = df0.loc[:nois_len, "severe_toxic"].as_matrix().astype('float16')

        data = np.append(data, data0)
        labels_sev_toxic = np.append(labels_sev_toxic, labels0_sev_toxic)

        print("sev_toxic")
        data_train, data_test, labels_train, labels_test = train_test_split(data, labels_sev_toxic, train_size=0.8)

        train_steps, train_batches = batch_iter(data_train, labels_train, batch_size)
        valid_steps, valid_batches = batch_iter(data_test, labels_test, batch_size)

        model.fit_generator(train_batches, train_steps,
                            epochs=epochs,
                            validation_data=valid_batches,
                            validation_steps=valid_steps)

        # /////////////////////save//////////////////////
        model.save_weights('w_sev_toxic.hdf5')
        json_string = model.to_json()
        open('model_sev_toxic.json', 'w').write(json_string)
    elif i == 2:
        ######################################obscene############################################

        df2 = df[(df["comment_text"].str.len() <=  input_len2) & (df["obscene"] > 0)]

        data = df2.loc[:, "comment_text"].as_matrix().astype('str')
        labels_obscene = df2.loc[:, "obscene"].as_matrix().astype('float16')

        df0 = df[(df["comment_text"].str.len() <=  input_len2) & (df["obscene"] == 0)]

        nois_len = len(data)

        data0 = df0.loc[:nois_len, "comment_text"].as_matrix().astype('str')
        labels0_obscene = df0.loc[:nois_len, "obscene"].as_matrix().astype('float16')

        data = np.append(data, data0)
        labels_obscene = np.append(labels_obscene, labels0_obscene)

        print("obscene")
        data_train, data_test, labels_train, labels_test = train_test_split(data, labels_obscene, train_size=0.8)

        train_steps, train_batches = batch_iter(data_train, labels_train, batch_size)
        valid_steps, valid_batches = batch_iter(data_test, labels_test, batch_size)

        model.fit_generator(train_batches, train_steps,
                            epochs=epochs,
                            validation_data=valid_batches,
                            validation_steps=valid_steps)

        # /////////////////////save//////////////////////
        json_string = model.to_json()
        open('model_obscene.json', 'w').write(json_string)
        model.save_weights('w_obscene.hdf5')
    elif i == 3:
        ######################################threat############################################

        df2 = df[(df["comment_text"].str.len() <= input_len2) & (df["threat"] > 0)]

        data = df2.loc[:, "comment_text"].as_matrix().astype('str')
        labels_threat = df2.loc[:, "threat"].as_matrix().astype('float16')

        df0 = df[(df["comment_text"].str.len() <= input_len2) & (df["threat"] == 0)]

        nois_len = len(data)

        data0 = df0.loc[:nois_len, "comment_text"].as_matrix().astype('str')
        labels0_threat = df0.loc[:nois_len, "threat"].as_matrix().astype('float16')


        data = np.append(data, data0)
        labels_threat = np.append(labels_threat, labels0_threat)

        print("threat")
        data_train, data_test, labels_train, labels_test = train_test_split(data, labels_threat, train_size=0.8)

        train_steps, train_batches = batch_iter(data_train, labels_train, batch_size)
        valid_steps, valid_batches = batch_iter(data_test, labels_test, batch_size)

        model.fit_generator(train_batches, train_steps,
                            epochs=epochs,
                            validation_data=valid_batches,
                            validation_steps=valid_steps)

        # /////////////////////save//////////////////////
        json_string = model.to_json()
        open('model_threat.json', 'w').write(json_string)
        model.save_weights('w_threat.hdf5')
    elif i == 4:
        ######################################insult############################################

        df2 = df[(df["comment_text"].str.len() <= input_len2) & (df["insult"] > 0)]

        data = df2.loc[:, "comment_text"].as_matrix().astype('str')
        labels_insult = df2.loc[:, "insult"].as_matrix().astype('float16')

        df0 = df[(df["comment_text"].str.len() <= input_len2) & (df["insult"] == 0)]

        nois_len = len(data)

        data0 = df0.loc[:nois_len, "comment_text"].as_matrix().astype('str')
        labels0_insult = df0.loc[:nois_len, "insult"].as_matrix().astype('float16')

        data = np.append(data, data0)
        labels_insult = np.append(labels_insult, labels0_insult)
        print("insult")
        data_train, data_test, labels_train, labels_test = train_test_split(data, labels_insult, train_size=0.8)

        train_steps, train_batches = batch_iter(data_train, labels_train, batch_size)
        valid_steps, valid_batches = batch_iter(data_test, labels_test, batch_size)

        model.fit_generator(train_batches, train_steps,
                            epochs=epochs,
                            validation_data=valid_batches,
                            validation_steps=valid_steps)

        # /////////////////////save//////////////////////

        json_string = model.to_json()
        open('model_insult.json', 'w').write(json_string)
        model.save_weights('w_insult.hdf5')
    elif i == 5:
        ######################################id_hate############################################

        df2 = df[(df["comment_text"].str.len() <= input_len2) & (df["identity_hate"] > 0)]

        data = df2.loc[:, "comment_text"].as_matrix().astype('str')
        labels_id_hate = df2.loc[:, "identity_hate"].as_matrix().astype('float16')

        df0 = df[(df["comment_text"].str.len() <= input_len2) & (df["identity_hate"] == 0)]

        nois_len = len(data)

        data0 = df0.loc[:nois_len, "comment_text"].as_matrix().astype('str')
        labels0_id_hate = df0.loc[:nois_len, "identity_hate"].as_matrix().astype('float16')

        data = np.append(data, data0)
        labels_id_hate = np.append(labels_id_hate, labels0_id_hate)
        print("id_hate")
        data_train, data_test, labels_train, labels_test = train_test_split(data, labels_id_hate, train_size=0.9)

        train_steps, train_batches = batch_iter(data_train, labels_train, batch_size)
        valid_steps, valid_batches = batch_iter(data_test, labels_test, batch_size)

        model.fit_generator(train_batches, train_steps,
                            epochs=epochs,
                            validation_data=valid_batches,
                            validation_steps=valid_steps)

        # /////////////////////save//////////////////////

        json_string = model.to_json()
        open('model_id_hate.json', 'w').write(json_string)
        model.save_weights('w_id_hate.hdf5')

