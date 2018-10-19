import os
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from skimage.io import imread
from skimage.transform import rescale
from scipy.misc import imresize
import matplotlib.pyplot as plt
from skimage.segmentation import mark_boundaries
# from skimage.util.montage import montage2d as montage
from skimage.morphology import binary_opening, disk
from sklearn.model_selection import train_test_split
from skimage.morphology import label
from keras.preprocessing.image import ImageDataGenerator
from keras import models, layers
import keras.backend as K
from keras.optimizers import Adam
from keras.losses import binary_crossentropy
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau
from tqdm import tqdm

import gc; gc.enable()


def IoU(y_true, y_pred):
    eps = 1e-6
    if np.max(y_true) == 0.0:
        return IoU(1-y_true, 1-y_pred) ## empty image; calc IoU of zeros
    intersection = K.sum(y_true * y_pred, axis=[1,2,3])
    union = K.sum(y_true, axis=[1,2,3]) + K.sum(y_pred, axis=[1,2,3]) - intersection
    return -K.mean( (intersection + eps) / (union + eps), axis=0)


def dice_coef(y_true, y_pred):
    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)
    intersection = K.sum(y_true * y_pred)
    return 2.0 * intersection / (K.sum(y_true) + K.sum(y_pred) + 1)


def dice_coef_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)




def tversky_index(y_true, y_pred):
    ALPHA = 0.7  # 0～1.0の値、Precision重視ならALPHAを大きくする
    BETA = 1.0 - ALPHA  # 0～1.0の値、Recall重視ならALPHAを小さくする
    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)
    intersection = K.sum(y_true * y_pred)
    false_positive = K.sum((1.0 - y_true) * y_pred)
    false_negative = K.sum(y_true * (1.0 - y_pred))
    return intersection / (intersection + ALPHA*false_positive + BETA*false_negative)


def tversky_loss(y_true, y_pred):
    return 1.0 - tversky_index(y_true, y_pred)



