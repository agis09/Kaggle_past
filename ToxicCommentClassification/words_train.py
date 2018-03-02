import numpy as np
import pandas as pd
from keras.models import Model
from keras.layers import Dense, Embedding, Input,GRU
from keras.layers import Conv1D, MaxPooling1D, GlobalMaxPool1D, Dropout, concatenate,Flatten
from keras.preprocessing import text, sequence
from sklearn.model_selection import train_test_split
from keras.optimizers import Adam
from keras import backend as K
from keras.layers.normalization import BatchNormalization

from keras.callbacks import EarlyStopping, ModelCheckpoint


max_features = 30000
maxlen = 100

embed_size = 200
##########train_data###########

df = pd.read_csv("train_preprocessed.csv", encoding="utf-8")
df2 = df[(df["toxic"] > 0) | (df["severe_toxic"] > 0) | (df["obscene"] > 0) | \
        (df["threat"] > 0) | (df["insult"] > 0) | (df["identity_hate"] > 0)]

data = df2.loc[:, "comment_text"].fillna("fillna")
data = data.as_matrix().astype('str')
labels = df2.loc[:, "toxic":"identity_hate"]
labels = labels.as_matrix().astype('float16')

np.random.permutation(df.index)
df0 = df[(df["toxic"] == 0) & (df["severe_toxic"] == 0) & (df["obscene"] == 0) & \
        (df["threat"] == 0) & (df["insult"] == 0) & (df["identity_hate"] == 0)].reset_index(drop=True)

data0 = df0.loc[:,"comment_text"].fillna("fillna")
data0 = data0.as_matrix().astype('str')
labels0 = df0.loc[:, "toxic":"identity_hate"]
labels0 = labels0.as_matrix().astype('float16')


leng = len(labels)+len(labels0)
list_sentences_train = np.append(data, data0)
y = np.append(labels, labels0)
y_train = np.reshape(y, (leng, 6))

#######test_data#########

df2 = pd.read_csv("test_preprocessed.csv", encoding="utf-8")

list_sentences_test = df2.loc[:, "comment_text"].fillna("fillna").as_matrix().astype('str')

list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

tokenizer = text.Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(list_sentences_train)+list(list_sentences_test))

X_train = tokenizer.texts_to_sequences(list_sentences_train)
X_test = tokenizer.texts_to_sequences(list_sentences_test)
x_train = sequence.pad_sequences(X_train, maxlen=maxlen)
x_test = sequence.pad_sequences(X_test, maxlen=maxlen)


def get_coefs(word, *arr): return word, np.asarray(arr, dtype='float32')


embeddings_index = dict(get_coefs(*o.strip().split()) for o in open("glove_6B_200d.txt", encoding="utf-8"))

all_embs = np.stack(embeddings_index.values())
emb_mean, emb_std = all_embs.mean(), all_embs.std()

word_index = tokenizer.word_index
nb_words = min(max_features, len(word_index))
embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
for word, i in word_index.items():
    if i >= max_features: continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None: embedding_matrix[i] = embedding_vector


def weight_variable(shape):
    return K.truncated_normal(shape, stddev=0.01)


def cnn_rnn():
    inp = Input(shape=(maxlen, ))
    main = Embedding(max_features, embed_size,weights=[embedding_matrix])(inp)
    main = Dropout(0.2)(main)
    main = Conv1D(filters=32, kernel_size=2, padding='same', activation='relu')(main)
    main = MaxPooling1D(pool_size=2)(main)
    main = Conv1D(filters=32, kernel_size=2, padding='same', activation='relu')(main)
    main = MaxPooling1D(pool_size=2)(main)

    main = GRU(32)(main)
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

[X_tra, X_val, y_tra, y_val] = train_test_split(x_train, y_train, train_size=0.95, random_state=233)

batch_size = 128
epochs = 5

hist = model.fit(X_tra, y_tra, batch_size=batch_size, epochs=epochs, validation_data=(X_val, y_val),
                 verbose=1)

model.save_weights("test.h5")
model.save('words_model.h5')


##############################predict##########################

test = pd.read_csv("test_preprocessed.csv",encoding="utf-8")
list_sentences_test = test["comment_text"].fillna("unknown").values


# test data
list_tokenized_test = tokenizer.texts_to_sequences(list_sentences_test)
X_te = sequence.pad_sequences(list_tokenized_test, maxlen=maxlen)

model.load_weights("test.h5")

y_pred = model.predict(x_test, batch_size=1024)

sample_submission = pd.read_csv("sample_submission.csv")
sample_submission[list_classes] = y_pred
sample_submission.to_csv("predictions.csv", index=False)
