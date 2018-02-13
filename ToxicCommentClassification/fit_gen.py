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

df = pd.read_csv("train.csv", encoding="utf-8")
df2 = df[(df["toxic"] > 0) | (df["severe_toxic"] > 0) | (df["obscene"] > 0) | \
        (df["threat"] > 0) | (df["insult"] > 0) | (df["identity_hate"] > 0)]

data = df2.loc[:, "comment_text"]
data = data.as_matrix().astype('str')
labels = df2.loc[:, "toxic":"identity_hate"]
labels = labels.as_matrix().astype('float16')

df0 = df[(df["toxic"] == 0) | (df["severe_toxic"] == 0) | (df["obscene"] == 0) | \
        (df["threat"] == 0) | (df["insult"] == 0) | (df["identity_hate"] == 0)]

data0 = df0.loc[:len(data)//6,"comment_text"]
data0 = data0.as_matrix().astype('str')
labels0 = df0.loc[:len(data)//6, "toxic":"identity_hate"]
labels0 = labels0.as_matrix().astype('float16')

np.append(data, data0)
np.append(labels,labels0)



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


data_train, data_test, labels_train, labels_test = train_test_split(data, labels, train_size=0.8)



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
n_out = 6
kernel_size = [7, 7, 3, 3, 3, 3]
nb_filter = 256
dense_units =[1024, 1024]
keep_prob = 0.5


model = build_model(kernel_sizes=kernel_size,
                    dense_units=dense_units,
                    vocab_size=27,
                    nb_filter=nb_filter,
                    nb_class=6,
                    keep_prob=keep_prob,
                    maxlen=input_len)

model.compile(loss='categorical_crossentropy',
              optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999),
              metrics=['accuracy'])


batch_size = 30

train_steps, train_batches = batch_iter(data_train, labels_train, batch_size)
valid_steps, valid_batches = batch_iter(data_test, labels_test, batch_size)


hist = model.fit_generator(train_batches, train_steps,
                           epochs=3,
                           validation_data=valid_batches,
                           validation_steps=valid_steps)

# hist = model.fit(X_train,Y_train,batch_size=batch_size, epochs=50, validation_data=(X_test, Y_test))

# /////////////////////save//////////////////////
model.save_weights('t3.hdf5')