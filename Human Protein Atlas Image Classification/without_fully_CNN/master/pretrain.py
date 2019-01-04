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
from weighted_binary_crossentropy import weighted_binary_crossentropy


def step_decay_schedule(initial_lr, step_size, min_lr):
    '''
    Wrapper function to create a LearningRateScheduler with step decay schedule.
    '''

    def schedule(epoch):
        return initial_lr * (np.cos(np.pi * epoch / step_size) + 1) / 2 + min_lr

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs['lr'] = K.get_value(self.model.optimizer.lr)
        super().on_epoch_end(epoch, logs)

    return LearningRateScheduler(schedule)


def pretrain_with_lr(path_to_train, data_frame, save_dir, batch_size, shape, lr):
    """
    path_to_train : 訓練画像のディレクトリ
    data_frame : train.csv
    save_dir : 各種結果の出力先
    lr : adamの学習率
    """
    model = create_model((shape[0], shape[1], 1))

    model.compile(
        #loss=[weighted_binary_crossentropy],
        loss='binary_crossentropy',
        # optimizer=SGD(lr=1e-4,momentum=0.9),
        optimizer=Adam(lr=lr),
        metrics=['acc', f1])
    model.summary()

    checkpoint = ModelCheckpoint(str(save_dir.joinpath('next_base.model%f'%lr)), monitor='val_loss', verbose=1, save_best_only=True,
                                 save_weights_only=False, mode='min', period=1)

    #lr_sched = step_decay_schedule(initial_lr=1e-3, step_size=epochs_per_iter, min_lr=1e-4)

    use_multiprocessing = True  # DO NOT COMBINE MULTIPROCESSING WITH CACHE!
    workers = 16  # DO NOT COMBINE MULTIPROCESSING WITH CACHE!
    max_queue_size=32

    loss_list = []
    val_loss_list = []
    f1_list = []
    val_f1_list = []

    val_sample_num = np.floor(len(data_frame) * 0.1).astype(int)
    val_indices = np.random.choice(range(0,len(data_frame)),val_sample_num,replace=False)
    val_data = data_frame.iloc[val_indices,]
    pathsVal, labelsVal = DataSet.getValidationDataset(path_to_train, val_data, False)
    pd.DataFrame([pathsVal]).to_csv(str(save_dir.joinpath("validation_paths.csv")), mode='w', header=False, index=False)

    train_indices = np.setdiff1d(range(0,len(data_frame)), val_indices) #train_indicesはここではまだ昇順(未シャッフル)
    epochs_per_iter = 5
    iters = 10
    for i in range(iters):
        train_indices_shuffled = np.random.permutation(train_indices)
        train_data_shuffled = data_frame.iloc[train_indices_shuffled,]
        pathsTrain, labelsTrain= DataSet.getTrainDataset100(path_to_train, train_data_shuffled, False)
        pd.DataFrame([pathsTrain],index=[i]).to_csv(str(save_dir.joinpath("train_paths.csv")), mode='w' if i==0 else 'a', header=False, index=True)

        print(pathsTrain.shape, labelsTrain.shape, pathsVal.shape, labelsVal.shape)
        tg = ProteinDataGenerator(pathsTrain, labelsTrain, batch_size, shape, is_mask=True, use_cache=False, augment=True, shuffle=True, crop_shape=(512, 512))
        vg = ProteinDataGenerator(pathsVal, labelsVal, batch_size, shape,is_mask=True, use_cache=False, shuffle=False, crop_shape=(512,512))

        tb = TensorBoard(log_dir=str(save_dir.joinpath('tb_logs_from'+format(epochs_per_iter * i + 1)+'-'+format(epochs_per_iter * i + 1)))+'epochs', histogram_freq=0, batch_size=batch_size)
        hist = model.fit_generator(
            tg,steps_per_epoch=len(tg), validation_data=vg, validation_steps=8, epochs=epochs_per_iter,
            use_multiprocessing=use_multiprocessing, workers=workers, max_queue_size=max_queue_size, verbose=1, callbacks=[tb,checkpoint])

        loss_list.extend(np.array(hist.history["loss"]).flatten())
        val_loss_list.extend(np.array(hist.history["val_loss"]).flatten())
        f1_list.extend(np.array(hist.history["f1"]).flatten())
        val_f1_list.extend(np.array(hist.history["val_f1"]).flatten())
        #histories.append(hist)
    pd.DataFrame([loss_list,val_loss_list,f1_list,val_f1_list],index=['loss','val_loss','f1','val_f1']).to_csv(str(save_dir.joinpath("results.csv")), mode='w', header=False, index=True)


    # fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    # ax[0].set_title('loss')
    # ax[0].plot(np.linspace(1,50,50), loss_list, label="Train loss")
    # ax[0].plot(np.linspace(1,50,50), val_loss_list, label="Validation loss")
    # ax[1].set_title('acc')
    # ax[1].plot(np.linspace(1,50,50), f1_list, label="Train F1")
    # ax[1].plot(np.linspace(1,50,50), val_f1_list, label="Validation F1")
    # ax[0].legend()
    # ax[1].legend()
    # plt.savefig(str(save_dir.joinpath('fcnn_mask_model_Adam%f.png'%lr)))
    return 1
