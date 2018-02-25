import numpy as np
import pandas as pd
from keras.models import Model
from keras.layers import Dense, Embedding, Input,GRU
from keras.layers import Conv1D, MaxPooling1D, GlobalMaxPool1D, Dropout, concatenate
from keras.preprocessing import text, sequence
from sklearn.model_selection import train_test_split
from keras.optimizers import Adam

from keras.models import model_from_json


from keras.callbacks import EarlyStopping, ModelCheckpoint

max_features = 20000
maxlen = 300

CLASSES_LIST = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

df = pd.read_csv("train.csv", encoding="utf-8")
tokenizer = text.Tokenizer(num_words=max_features,
                           filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
                           lower=True,
                           split=" ",
                           char_level=False)


def cnn_rnn():
    embed_size = 256
    inp = Input(shape=(maxlen, ))
    main = Embedding(max_features, embed_size)(inp)
    main = Dropout(0.2)(main)
    main = Conv1D(filters=32, kernel_size=2, padding='same', activation='relu')(main)
    main = MaxPooling1D(pool_size=2)(main)
    main = Conv1D(filters=32, kernel_size=2, padding='same', activation='relu')(main)
    main = MaxPooling1D(pool_size=2)(main)
    main = GRU(32)(main)
    main = Dense(16, activation="relu")(main)
    main = Dense(1, activation="sigmoid")(main)
    model = Model(inputs=inp, outputs=main)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model


for i in range(6):
    model = cnn_rnn()
    if i==0:model.summary()
    model.compile(loss='binary_crossentropy',
                  optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999),
                  metrics=['accuracy'])

    #############################train#######################################
    np.random.permutation(df.index)

    df2 = df[(df[CLASSES_LIST[i]] > 0)]

    data = df2.loc[:, "comment_text"].fillna("unknown").as_matrix().astype('str')
    labels = df2.loc[:, CLASSES_LIST[i]].as_matrix().astype('float16')

    np.random.permutation(df.index)

    df0 = df[(df[CLASSES_LIST[i]] < 1)].reset_index(drop=True)

    data0 = df0.loc[:len(data), "comment_text"].fillna("unknown").as_matrix().astype('str')
    labels0 = df0.loc[:len(data), CLASSES_LIST[i]].as_matrix().astype('float16')

    list_sentences_train = np.append(data, data0)
    y = np.append(labels, labels0)

    tokenizer.fit_on_texts(list(list_sentences_train))

    # train data
    list_tokenized_train = tokenizer.texts_to_sequences(list_sentences_train)
    X_t = sequence.pad_sequences(list_tokenized_train, maxlen=maxlen)

    X_t_train, X_t_test, y_train, y_test = train_test_split(X_t, y, test_size=0.10)

    batch_size = 128
    epochs = 3

    print(CLASSES_LIST[i])
    model.fit(X_t_train, y_train,
              validation_data=(X_t_test, y_test),
              batch_size=batch_size,
              epochs=epochs,
              shuffle=True)

    model.save_weights("words_w_"+CLASSES_LIST[i]+".h5")
    json_string = model.to_json()
    open('words_model_' + CLASSES_LIST[i] + '.json', 'w').write(json_string)

    ##############################predict##########################

# for i in range(6):
    print(CLASSES_LIST[i])
    model = model_from_json(open('words_model_' + CLASSES_LIST[i] + '.json').read())

    model.compile(loss='binary_crossentropy',
                  optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999),
                  metrics=['accuracy'])

    test = pd.read_csv("test.csv", encoding="utf-8")
    list_sentences_test = test["comment_text"].fillna("unknown").values

    # test data
    list_tokenized_test = tokenizer.texts_to_sequences(list_sentences_test)
    X_te = sequence.pad_sequences(list_tokenized_test, maxlen=maxlen)

    model.load_weights("words_w_"+CLASSES_LIST[i]+".h5")

    y_test = model.predict(X_te, verbose=1)
    if i == 0 :
        sample_submission = pd.read_csv("sample_submission.csv",encoding="utf-8")
        sample_submission.to_csv("predictions.csv", index=False)
    predictions = pd.read_csv("predictions.csv")
    predictions[CLASSES_LIST[i]] = y_test
    predictions.to_csv("predictions.csv", index=False)

