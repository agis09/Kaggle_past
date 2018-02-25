import numpy as np
import pandas as pd
from keras.models import Model
from keras.layers import Dense, Embedding, Input,GRU
from keras.layers import Conv1D, MaxPooling1D, GlobalMaxPool1D, Dropout, concatenate,Flatten
from keras.preprocessing import text, sequence
from sklearn.model_selection import train_test_split
from keras.optimizers import Adam
from keras import backend as K

from keras.callbacks import EarlyStopping, ModelCheckpoint

max_features = 20000
maxlen = 300

df = pd.read_csv("train.csv", encoding="utf-8")
df2 = df[(df["toxic"] > 0) | (df["severe_toxic"] > 0) | (df["obscene"] > 0) | \
        (df["threat"] > 0) | (df["insult"] > 0) | (df["identity_hate"] > 0)]

data = df2.loc[:, "comment_text"].fillna("unknown")
data = data.as_matrix().astype('str')
labels = df2.loc[:, "toxic":"identity_hate"]
labels = labels.as_matrix().astype('float16')

np.random.permutation(df.index)
df0 = df[(df["toxic"] == 0) & (df["severe_toxic"] == 0) & (df["obscene"] == 0) & \
        (df["threat"] == 0) & (df["insult"] == 0) & (df["identity_hate"] == 0)].reset_index(drop=True)

data0 = df0.loc[:len(data)//2,"comment_text"].fillna("unknown")
data0 = data0.as_matrix().astype('str')
labels0 = df0.loc[:len(data)//2, "toxic":"identity_hate"]
labels0 = labels0.as_matrix().astype('float16')


# print(labels)
leng = len(labels)+len(labels0)
list_sentences_train = np.append(data, data0)
y = np.append(labels, labels0)
y = np.reshape(y, (leng, 6))
# print(y)


list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

tokenizer = text.Tokenizer(num_words=max_features,
                           filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
                           lower=True,
                           split=" ",
                           char_level=False
                           )
tokenizer.fit_on_texts(list(list_sentences_train))

# train data
list_tokenized_train = tokenizer.texts_to_sequences(list_sentences_train)
X_t = sequence.pad_sequences(list_tokenized_train, maxlen=maxlen)


def weight_variable(shape):
    return K.truncated_normal(shape, stddev=0.01)


def cnn_rnn():
    embed_size = 256
    inp = Input(shape=(maxlen, ))
    main = Embedding(max_features, embed_size)(inp)
    main = Dropout(0.2)(main)
    main = Conv1D(filters=32, kernel_size=2, padding='same', activation='relu')(main)
    main = MaxPooling1D(pool_size=2)(main)
    main = Conv1D(filters=32, kernel_size=2, padding='same', activation='relu')(main)
    main = MaxPooling1D(pool_size=2)(main)

    main = Dropout(0.25)(main)
    main = Conv1D(filters=32, kernel_size=2, padding='same', activation='relu')(main)
    main = MaxPooling1D(pool_size=2)(main)

    # main = GRU(32,init=weight_variable,return_sequences=True)(main)
    main = GRU(32, init=weight_variable,return_sequences=False)(main)
    main = Dense(16, activation="relu")(main)
    main = Dense(6, activation="sigmoid")(main)
    model = Model(inputs=inp, outputs=main)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model

model = cnn_rnn()
model.summary()

model.compile(loss='binary_crossentropy',
                  optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999),
                  metrics=['accuracy'])
#############################train#######################################

X_t_train, X_t_test, y_train, y_test = train_test_split(X_t, y, test_size = 0.10)

batch_size = 128
epochs = 5

model.fit(X_t_train, y_train,
          validation_data=(X_t_test, y_test),
          batch_size=batch_size,
          epochs=epochs,
          shuffle = True)

model.save_weights("test.h5")
model.save('words_model.h5')


##############################predict##########################

test = pd.read_csv("test.csv",encoding="utf-8")
list_sentences_test = test["comment_text"].fillna("unknown").values


# test data
list_tokenized_test = tokenizer.texts_to_sequences(list_sentences_test)
X_te = sequence.pad_sequences(list_tokenized_test, maxlen=maxlen)

model.load_weights("test.h5")

y_test = model.predict(X_te)
sample_submission = pd.read_csv("sample_submission.csv")
sample_submission[list_classes] = y_test
sample_submission.to_csv("predictions.csv", index=False)