import sys
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
#from matplotlib import pyplot as plt

from tensorflow import set_random_seed
from keras.models import Sequential, load_model
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau, TensorBoard
from keras.optimizers import Adam, SGD
from keras import regularizers, metrics
from keras import backend as K

from scipy import ndimage
import signal
from pathlib import Path

import DataSet
from ProteinDataGenerator import ProteinDataGenerator
from f1 import *
from model import *
from sklearn.metrics import f1_score as off1
import os
from weighted_binary_crossentropy import weighted_binary_crossentropy

DIR='../'

def getTestDataset():
    
    path_to_test = DIR + 'test/'
    data = pd.read_csv(DIR + 'sample_submission.csv')

    paths = []
    labels = []

    for name in data['Id']:
        y = np.ones(28)
        paths.append(os.path.join(path_to_test, name))
        labels.append(y)

    return np.array(paths), np.array(labels)


def train(path_to_train, data_frame, pretrained_weights, save_dir, batch_size, shape, lr, val_ratio, epochs):
    model = create_discriminant_model((shape[0], shape[1], 1), pretrained_weights)
    for layer in model.layers[:-2]:
        layer.trainable = False
    model.compile(
        loss=[weighted_binary_crossentropy],
        # loss='binary_crossentropy',
        # optimizer=SGD(lr=1e-4,momentum=0.9),
        optimizer=Adam(lr=lr),
        metrics=['acc', f1])
    model.summary()
    checkpoint = ModelCheckpoint(str(save_dir.joinpath('next_base.model%f'%lr)), monitor='val_loss', verbose=1, save_best_only=True,
                                 save_weights_only=False, mode='min', period=1)
    #lr_sched = step_decay_schedule(initial_lr=1e-3, step_size=epochs, min_lr=1e-4)

    use_multiprocessing = False  # DO NOT COMBINE MULTIPROCESSING WITH CACHE!
    workers = 1  # DO NOT COMBINE MULTIPROCESSING WITH CACHE!

    val_sample_num = np.floor(len(data_frame) * val_ratio).astype(int)
    val_indices = np.random.choice(range(0,len(data_frame)),val_sample_num,replace=False)
    val_data = data_frame.iloc[val_indices,]
    pathsVal, labelsVal = DataSet.getValidationDataset(path_to_train, val_data, False)
    pd.DataFrame([pathsVal]).to_csv(str(save_dir.joinpath("validation_paths.csv")), mode='w', header=False, index=False)

    train_indices = np.setdiff1d(range(0,len(data_frame)), val_indices) #train_indicesはここではまだ昇順(未シャッフル)
    train_indices = np.random.permutation(train_indices)
    train_data = data_frame.iloc[train_indices,]
    pathsTrain, labelsTrain= DataSet.getTrainDataset(path_to_train, train_data, False)
    pd.DataFrame([pathsTrain]).to_csv(str(save_dir.joinpath("train_paths.csv")), mode='w', header=False, index=False)

    print(pathsTrain.shape, labelsTrain.shape, pathsVal.shape, labelsVal.shape)
    tg = ProteinDataGenerator(pathsTrain, labelsTrain, batch_size, shape, is_mask=False, use_cache=False, augment=True, shuffle=True)
    vg = ProteinDataGenerator(pathsVal, labelsVal, batch_size, shape, is_mask=False, use_cache=False, shuffle=False)

    tb = TensorBoard(log_dir=str(save_dir.joinpath('tb_logs')), histogram_freq=0, batch_size=batch_size)

    hist = model.fit_generator(
        tg,steps_per_epoch=len(tg), validation_data=vg, validation_steps=8, epochs=epochs,
        use_multiprocessing=use_multiprocessing, workers=workers, verbose=1, callbacks=[tb,checkpoint])

    loss_list = np.array(hist.history["loss"]).flatten()
    val_loss_list = np.array(hist.history["val_loss"]).flatten()
    f1_list =np.array(hist.history["f1"]).flatten()
    val_f1_list = np.array(hist.history["val_f1"]).flatten()
    #histories.append(hist)
    pd.DataFrame([loss_list,val_loss_list,f1_list,val_f1_list],index=['loss','val_loss','f1','val_f1']).to_csv(str(save_dir.joinpath("results.csv")), mode='w', header=False, index=True)

    # fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    # ax[0].set_title('loss')
    # ax[0].plot(np.linspace(1,epochs,epochs), loss_list, label="Train loss")
    # ax[0].plot(np.linspace(1,epochs,epochs), val_loss_list, label="Validation loss")
    # ax[1].set_title('acc')
    # ax[1].plot(np.linspace(1,epochs,epochs), f1_list, label="Train F1")
    # ax[1].plot(np.linspace(1,epochs,epochs), val_f1_list, label="Validation F1")
    # ax[0].legend()
    # ax[1].legend()
    # plt.savefig(str(save_dir.joinpath('fcnn_model_Adam%f.png'%lr)))

    print('#######################')   
    bestModel = load_model(str(save_dir.joinpath('next_base.model%f'%lr)) , custom_objects={'f1': f1 ,'weighted_binary_crossentropy':weighted_binary_crossentropy})#  'f1_loss': f1_loss})
    print('#######################')   
    """
    fullValGen = ProteinDataGenerator(pathsVal, labelsVal, batch_size, shape, is_mask=False,use_cache=False, shuffle=False)
    lastFullValPred = np.empty((0, 28))
    lastFullValLabels = np.empty((0, 28))
    for i in tqdm(range(len(fullValGen))):
        im, lbl = fullValGen[i]
        scores = bestModel.predict(im)
        lastFullValPred = np.append(lastFullValPred, scores, axis=0)
        lastFullValLabels = np.append(lastFullValLabels, lbl, axis=0)
    print(lastFullValPred.shape, lastFullValLabels.shape)

    rng = np.arange(0, 1, 0.001)
    f1s = np.zeros((rng.shape[0], 28))
    for j, t in enumerate(tqdm(rng)):
        for i in range(28):
            p = np.array(lastFullValPred[:, i] > t, dtype=np.int8)
            scoref1 = off1(lastFullValLabels[:, i], p, average='binary')
            f1s[j, i] = scoref1
    print('Individual F1-scores for each class:')
    print(np.max(f1s, axis=0))
    print('Macro F1-score CV =', np.mean(np.max(f1s, axis=0)))

    # plt.plot(rng, f1s)
    T = np.empty(28)
    for i in range(28):
        T[i] = rng[np.where(f1s[:, i] == np.max(f1s[:, i]))[0][0]]
        if T[i] < 0.01:
            T[i] = 0.01
    print('Probability threshold maximizing CV F1-score for each class:')
    print(T)
    
    pathsTest, labelsTest = getTestDataset()
    testg = ProteinDataGenerator(pathsTest, labelsTest, batch_size, shape,is_mask=False, shuffle=False, use_cache=False, augment=False)
    submit = pd.read_csv(DIR + '/sample_submission.csv')

    P = np.zeros((pathsTest.shape[0], 28))
    for i in tqdm(range(len(testg))):
        images, labels = testg[i]
        score = bestModel.predict(images)
        P[i * batch_size:i * batch_size + score.shape[0]] = score

    PP = np.array(P)
    prediction = []

    for row in tqdm(range(submit.shape[0])):
        str_label = ''
        for col in range(PP.shape[1]):
            if (PP[row, col] < T[col]):
                str_label += ''
            else:
                str_label += str(col) + ' '
        prediction.append(str_label.strip())
    """
    pathsTest, labelsTest = getTestDataset()
    testg = ProteinDataGenerator(pathsTest, labelsTest, batch_size, shape,is_mask=False, shuffle=False, use_cache=False, augment=False)
    submit = pd.read_csv(DIR + '/sample_submission.csv')

    P = np.zeros((pathsTest.shape[0], 28))
    for i in tqdm(range(len(testg))):
        images, labels = testg[i]
        score = bestModel.predict(images)
        P[i * batch_size:i * batch_size + score.shape[0]] = score

    PP = np.array(P)
    prediction = []

    for row in tqdm(range(submit.shape[0])):
        str_label = ''
        for col in range(PP.shape[1]):
            if (PP[row, col] < 0.5):
                str_label += ''
            else:
                str_label += str(col) + ' '
        prediction.append(str_label.strip())
    submit['Predicted'] = np.array(prediction)
    submit.to_csv('4channels_cnn.csv', index=False)
    print(submit.head())
    return 1
