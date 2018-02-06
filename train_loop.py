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

df = pd.read_csv("train.csv", encoding="utf-8")
for loop in range(0,64):
    if loop == 63:
        e = 1597
    else :
        e = 2500
    s = loop*2500
    x = df.loc[s:s+e, "comment_text"]

    y = df.loc[s:s+e, "toxic":"identity_hate"]
    y = y.as_matrix().astype('float16')


    def make_one_hot(char):
        index = ord(char) - ord("a")
        value = np.zeros(26)
        if index >= 0 and index <= 25:
            value[index] = 1.0
        return value


    def text_to_matrix(text):
        matrix = np.array([make_one_hot(char) for char in text])
        return matrix


    data = np.array([])
    for words in x:
        words = words.lower()
        tmp = np.array([])
        cnt = 1014
        for word in words:
            tmp = np.append(tmp, text_to_matrix(word))
            cnt -= 1
            if cnt == 0:
                break
        if cnt > 0:
            for i in range(0, cnt):
                tmp = np.append(tmp, np.zeros(26))
        tmp = np.reshape(tmp, (1014, 26))
        data = np.append(data, tmp)
    data = np.reshape(data, (len(x), 1014, 26))

    np.save('data'+str(loop), data)
    np.save('y'+str(loop), y)
    """
    data = np.load('data.npy')
    y = np.load('y.npy')
    """
    X_train, X_test, Y_train, Y_test = train_test_split(data, y, train_size=0.8)


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
    n_out = len(y[0])
    kernel_size = [7, 7, 3, 3, 3, 3]
    nb_filter = 256
    dense_units = [1024, 1024]
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
    if loop != 0:
        model.load_weights('t1.hdf5')

    batch_size = 50

    hist = model.fit(X_train, Y_train, batch_size=batch_size, epochs=50, validation_data=(X_test, Y_test))

    # /////////////////////save//////////////////////
    model.save_weights('t1.hdf5')

    plt.clf()
    plt.plot(hist.history['acc'], label="accuracy")
    plt.plot(hist.history['val_acc'], label="val_acc")
    plt.title('model_acc')
    plt.xlabel('epoch')
    plt.ylabel('acc')
    plt.legend(loc="lower right")
    plt.savefig('ACC'+str(loop))

    plt.clf()
    plt.plot(hist.history['loss'], label="loss")
    plt.plot(hist.history['val_loss'], label="val_loss")
    plt.title('model_loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(loc="lower right")
    plt.savefig('Loss'+str(loop))
