import sys
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from matplotlib import pyplot as plt

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

def train(path_to_train, data_frame, pretrained_weights, save_dir, batch_size, shape, lr, val_ratio, epochs):
    model = create_discriminant_model((shape[0], shape[1], 1), pretrained_weights)
    for layer in model.layers[:-2]:
        layer.is_trainable = False
    model.compile(
        # loss=[weighted_binary_crossentropy],
        loss='binary_crossentropy',
        # optimizer=SGD(lr=1e-4,momentum=0.9),
        optimizer=Adam(lr=lr),
        metrics=['acc', f1])
    model.summary()
    checkpoint = ModelCheckpoint(str(save_dir.joinpath('next_base.model%f.epoch{epoch:02d}'%lr)), monitor='val_loss', verbose=1, save_best_only=True,
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
    tg = ProteinDataGenerator(pathsTrain, labelsTrain, batch_size, shape, is_mask=False, use_cache=True, augment=True, shuffle=True)
    vg = ProteinDataGenerator(pathsVal, labelsVal, batch_size, shape, is_mask=False, use_cache=True, shuffle=False)

    tb = TensorBoard(log_dir=str(save_dir.joinpath('tb_logs')), histogram_freq=0, batch_size=batch_size)

    hist = model.fit_generator(
        tg,steps_per_epoch=len(tg), validation_data=vg, validation_steps=8, epochs=epochs,
        use_multiprocessing=use_multiprocessing, workers=workers, verbose=1, callbacks=[tb,checkpoint])

    loss_list = hist.history["loss"]
    val_loss_list = hist.history["val_loss"]
    f1_list = hist.history["f1"]
    val_f1_list = hist.history["val_f1"]
    #histories.append(hist)
    pd.DataFrame([loss_list,val_loss_list,f1_list,val_f1_list],index=['loss','val_loss','f1','val_f1']).to_csv(str(res_dir.joinpath("results.csv")), mode='w', header=False, index=True)

    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    ax[0].set_title('loss')
    ax[0].plot(np.linspace(1,50,50), loss_list, label="Train loss")
    ax[0].plot(np.linspace(1,50,50), val_loss_list, label="Validation loss")
    ax[1].set_title('acc')
    ax[1].plot(np.linspace(1,50,50), f1_list, label="Train F1")
    ax[1].plot(np.linspace(1,50,50), val_f1_list, label="Validation F1")
    ax[0].legend()
    ax[1].legend()
    plt.savefig(str(save_dir.joinpath('fcnn_model_Adam%f.png'%lr)))

    return 1
