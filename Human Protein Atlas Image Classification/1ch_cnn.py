# from https://www.kaggle.com/rejpalcz/gapnet-pl-lb-0-385
import sys
import numpy as np
import keras
from keras.utils import Sequence
from PIL import Image
from matplotlib import pyplot as plt
import pandas as pd
from tqdm import tqdm
import os
# import imgaug as ia
# from imgaug import augmenters as iaa
import cv2
import random

BATCH_SIZE = 16
SEED = 777
SHAPE = (512, 512, 4)
DIR = '../'
VAL_RATIO = 0.2  # 20 % as validation
THRESHOLD = 0.5  # due to different cost of True Positive vs False Positive, this is the probability threshold to predict the class as 'yes'


# ia.seed(SEED)

def getTrainDataset():
    path_to_train = DIR + 'train/'
    data = pd.read_csv(DIR + 'train.csv')

    paths = []
    labels = []

    for name, lbl in zip(data['Id'], data['Target'].str.split(' ')):
        y = np.zeros(28)
        for key in lbl:
            y[int(key)] = 1
        paths.append(os.path.join(path_to_train, name))
        labels.append(y)

    return np.array(paths), np.array(labels)


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


# credits: https://github.com/keras-team/keras/blob/master/keras/utils/data_utils.py#L302
# credits: https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly

class ProteinDataGenerator(keras.utils.Sequence):

    def __init__(self, paths, labels, batch_size, shape, shuffle=False, use_cache=False, augment=False):
        self.paths, self.labels = paths, labels
        self.batch_size = batch_size
        self.shape = shape
        self.shuffle = shuffle
        self.use_cache = use_cache
        self.augment = augment
        if use_cache == True:
            self.cache = np.zeros((paths.shape[0], shape[0], shape[1], shape[2]), dtype=np.float16)
            self.is_cached = np.zeros((paths.shape[0]))
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.paths) / float(self.batch_size)))

    def __getitem__(self, idx):
        indexes = self.indexes[idx * self.batch_size: (idx + 1) * self.batch_size]

        paths = self.paths[indexes]
        X = np.zeros((paths.shape[0], self.shape[0], self.shape[1], self.shape[2]))
        # Generate data
        if self.use_cache == True:
            X = self.cache[indexes]
            for i, path in enumerate(paths[np.where(self.is_cached[indexes] == 0)]):
                image = self.__load_image(path)
                self.is_cached[indexes[i]] = 1
                self.cache[indexes[i]] = image
                X[i] = image
        else:
            for i, path in enumerate(paths):
                X[i] = self.__load_image(path)

        y = self.labels[indexes]

        if self.augment == True:
            """
            seq = iaa.Sequential([
            iaa.OneOf([
                    iaa.Fliplr(0.5), # horizontal flips
                    iaa.Crop(percent=(0, 0.1)), # random crops
                    # Small gaussian blur with random sigma between 0 and 0.5.
                    # But we only blur about 50% of all images.
                    iaa.Sometimes(0.5,
                        iaa.GaussianBlur(sigma=(0, 0.5))
                    ),
                    # Strengthen or weaken the contrast in each image.
                    iaa.ContrastNormalization((0.75, 1.5)),
                    # Add gaussian noise.
                    # For 50% of all images, we sample the noise once per pixel.
                    # For the other 50% of all images, we sample the noise per pixel AND
                    # channel. This can change the color (not only brightness) of the
                    # pixels.
                    iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5),
                    # Make some images brighter and some darker.
                    # In 20% of all cases, we sample the multiplier once per channel,
                    # which can end up changing the color of the images.
                    iaa.Multiply((0.8, 1.2), per_channel=0.2),
                    # Apply affine transformations to each image.
                    # Scale/zoom them, translate/move them, rotate them and shear them.
                    iaa.Affine(
                        scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                        translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
                        rotate=(-180, 180),
                        shear=(-8, 8)
                    )
                ])], random_order=True)

            X = np.concatenate((X, seq.augment_images(X), seq.augment_images(X), seq.augment_images(X)), 0)
            y = np.concatenate((y, y, y, y), 0)
            """
        # return X, y
        # return [np.reshape(X[:,:,:,i],(X.shape[0],SHAPE[0],SHAPE[1],1)) for i in range(4)],y
        return [np.reshape(X[:, :, :, 1], (X.shape[0], SHAPE[0], SHAPE[1], 1))], y

    def on_epoch_end(self):

        # Updates indexes after each epoch
        self.indexes = np.arange(len(self.paths))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __iter__(self):
        """Create a generator that iterate over the Sequence."""
        for item in (self[i] for i in range(len(self))):
            yield item

    def __load_image(self, path):
        R = Image.open(path + '_red.png')
        G = Image.open(path + '_green.png')
        B = Image.open(path + '_blue.png')
        Y = Image.open(path + '_yellow.png')

        im = np.stack((
            np.array(R),
            np.array(G),
            np.array(B),
            np.array(Y)), -1)

        im = cv2.resize(im, (SHAPE[0], SHAPE[1]))
        im = np.divide(im, 255)
        if self.augment:
            if random.randint(0, 1):
                im = im[:, ::-1, :]
            if random.randint(0, 1):
                im = im[::-1, :, :]
        return im

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, load_model, Model
from keras.layers import Activation, Dropout, Flatten, Dense, Input, Conv2D, MaxPooling2D, BatchNormalization, Concatenate, ReLU, LeakyReLU, GlobalAveragePooling2D
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau
from keras import metrics
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras import backend as K
import keras
import tensorflow as tf

from tensorflow import set_random_seed
set_random_seed(SEED)


# credits: https://www.kaggle.com/guglielmocamporese/macro-f1-score-keras

def f1(y_true, y_pred):
    # y_pred = K.round(y_pred)
    # y_pred = K.cast(K.greater(K.clip(y_pred, 0, 1), THRESHOLD), K.floatx())
    tp = K.sum(K.cast(y_true * y_pred, 'float'), axis=0)
    tn = K.sum(K.cast((1 - y_true) * (1 - y_pred), 'float'), axis=0)
    fp = K.sum(K.cast((1 - y_true) * y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true * (1 - y_pred), 'float'), axis=0)

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    f1 = 2 * p * r / (p + r + K.epsilon())
    f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)
    return K.mean(f1)


def f1_loss(y_true, y_pred):
    # y_pred = K.cast(K.greater(K.clip(y_pred, 0, 1), THRESHOLD), K.floatx())
    tp = K.sum(K.cast(y_true * y_pred, 'float'), axis=0)
    tn = K.sum(K.cast((1 - y_true) * (1 - y_pred), 'float'), axis=0)
    fp = K.sum(K.cast((1 - y_true) * y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true * (1 - y_pred), 'float'), axis=0)

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    f1 = 2 * p * r / (p + r + K.epsilon())
    f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)
    return 1 - K.mean(f1)


from keras.layers.core import Lambda
from keras import regularizers
from keras.layers import concatenate, Flatten, GlobalAveragePooling2D, Add, AveragePooling2D, multiply, Reshape
from keras import backend as K


def se_block(ch, layer, ratio=8):
    z = GlobalAveragePooling2D()(layer)
    x = Dense(ch // ratio, activation='relu')(z)
    x = Dense(ch, activation='sigmoid')(x)
    x = Reshape((1, 1, ch))(x)
    layer = multiply([layer, x])

    return layer


def res_block(ch, layer):
    layer = Conv2D(ch, (3, 3))(layer)
    layer = BatchNormalization()(layer)
    layer = ReLU()(layer)
    skip = layer

    layer = BatchNormalization()(layer)
    layer = Conv2D(ch, (3, 3), padding='same')(layer)
    layer = BatchNormalization()(layer)
    layer = ReLU()(layer)
    layer = Conv2D(ch, (3, 3), padding='same')(layer)
    layer = BatchNormalization()(layer)
    layer = ReLU()(layer)

    layer = se_block(ch, layer)
    layer = Add()([layer, skip])

    skip = layer

    layer = BatchNormalization()(layer)
    layer = Conv2D(ch, (3, 3), padding='same')(layer)
    layer = BatchNormalization()(layer)
    layer = ReLU()(layer)
    layer = Conv2D(ch, (3, 3), padding='same')(layer)
    layer = BatchNormalization()(layer)
    layer = ReLU()(layer)

    layer = se_block(ch, layer)
    layer = Add()([layer, skip])

    return layer


def ch_CNN(layer):
    # layer=BatchNormalization()(layer)
    """
    layer=Conv2D(16,(3,3))(layer)
    layer=se_block(16,layer)
    layer=BatchNormalization()(layer)
    layer=ReLU()(layer)
    layer=MaxPooling2D(pool_size=(2,2))(layer)

    layer=res_block(16,layer)
    layer=MaxPooling2D(pool_size=(2,2))(layer)

    layer=res_block(32,layer)
    layer=MaxPooling2D(pool_size=(2,2))(layer)

    layer=res_block(64,layer)
    layer=MaxPooling2D(pool_size=(2,2))(layer)
    """

    layer = Conv2D(16, (2, 2))(layer)
    # layer=BatchNormalization()(layer)
    layer = ReLU()(layer)
    layer = MaxPooling2D(pool_size=(2, 2))(layer)
    layer = se_block(16, layer)
    layer = Conv2D(32, (3, 3))(layer)
    # layer=BatchNormalization()(layer)
    layer = ReLU()(layer)
    layer = MaxPooling2D(pool_size=(2, 2))(layer)
    layer = se_block(32, layer)
    layer = Conv2D(64, (3, 3))(layer)
    # layer=BatchNormalization()(layer)
    layer = ReLU()(layer)
    layer = MaxPooling2D(pool_size=(2, 2))(layer)
    layer = se_block(64, layer)

    return layer


def create_model(input_shape):
    ch1 = Input(input_shape)
    # ch2 = Input(input_shape)
    # ch3 = Input(input_shape)
    # ch4 = Input(input_shape)

    x = ch_CNN(ch1)
    # x2=ch_CNN(ch2)
    # x3=ch_CNN(ch3)
    # x4=ch_CNN(ch4)

    # x=concatenate([x1,x2,x3,x4])
    # x=Add()([x1,x2,x3,x4])
    # x=se_block(64,x)

    x = Conv2D(64, (3, 3))(x)
    # x=BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(28, (3, 3))(x)
    x = Activation('relu')(x)
    x = GlobalAveragePooling2D()(x)
    x = Activation('sigmoid')(x)

    # model = Model([ch1,ch2,ch3,ch4], x)
    model = Model(ch1, x)

    return model


import tensorflow as tf
import keras.backend.tensorflow_backend as tfb
from keras.optimizers import SGD

POS_WEIGHT = 10  # multiplier for positive targets, needs to be tuned

def weighted_binary_crossentropy(target, output):
    """
    Weighted binary crossentropy between an output tensor
    and a target tensor. POS_WEIGHT is used as a multiplier
    for the positive targets.

    Combination of the following functions:
    * keras.losses.binary_crossentropy'binary_crossentropy'
    * keras.backend.tensorflow_backend.binary_crossentropy
    * tf.nn.weighted_cross_entropy_with_logits
    """
    # transform back to logits
    _epsilon = tfb._to_tensor(tfb.epsilon(), output.dtype.base_dtype)
    output = tf.clip_by_value(output, _epsilon, 1 - _epsilon)
    output = tf.log(output / (1 - output))
    # compute weighted loss
    loss = tf.nn.weighted_cross_entropy_with_logits(targets=target,
                                                    logits=output,
                                                    pos_weight=POS_WEIGHT)
    return tf.reduce_mean(loss, axis=-1)

model = create_model((SHAPE[0],SHAPE[1],1))


#############################
# fine_tuning
# model = load_model('./base.model', custom_objects={'f1': f1,'weighted_binary_crossentropy':weighted_binary_crossentropy}) #, 'f1_loss': f1_loss})
#############################



model.compile(
    loss=[weighted_binary_crossentropy],
    optimizer=Adam(lr=1e-3),
    # optimizer=SGD(lr=1e-2,momentum=0.9),
    metrics=['acc',f1])
model.summary()

paths, labels = getTrainDataset()

# divide to
keys = np.arange(paths.shape[0], dtype=np.int)
np.random.seed(SEED)
np.random.shuffle(keys)
lastTrainIndex = int((1 - VAL_RATIO) * paths.shape[0])

pathsTrain = paths[0:lastTrainIndex]
labelsTrain = labels[0:lastTrainIndex]
pathsVal = paths[lastTrainIndex:]
labelsVal = labels[lastTrainIndex:]

print(paths.shape, labels.shape)
print(pathsTrain.shape, labelsTrain.shape, pathsVal.shape, labelsVal.shape)

tg = ProteinDataGenerator(pathsTrain, labelsTrain, BATCH_SIZE, SHAPE, use_cache=False, augment=True, shuffle=True)
vg = ProteinDataGenerator(pathsVal, labelsVal, BATCH_SIZE, SHAPE, use_cache=False, shuffle=False)

# https://keras.io/callbacks/#modelcheckpoint
checkpoint = ModelCheckpoint('./base.model', monitor='val_loss', verbose=1, save_best_only=True,
                             save_weights_only=False, mode='min', period=1)
# checkpoint = ModelCheckpoint('./best.model', monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False, mode='min', period=1)
# reduceLROnPlato = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1, mode='min')

from keras.callbacks import TensorBoard


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

epochs = 100
lr_sched = step_decay_schedule(initial_lr=1e-2, step_size=epochs,min_lr=1e-5)

use_multiprocessing = False # DO NOT COMBINE MULTIPROCESSING WITH CACHE!
workers = 1 # DO NOT COMBINE MULTIPROCESSING WITH CACHE!

hist = model.fit_generator(
    tg,
    steps_per_epoch=len(tg),
    validation_data=vg,
    validation_steps=8,
    epochs=epochs,
    use_multiprocessing=use_multiprocessing,
    workers=workers,
    verbose=1,
    callbacks=[checkpoint])
    # callbacks=[checkpoint,lr_sched])

fig, ax = plt.subplots(1, 2, figsize=(15,5))
ax[0].set_title('loss')
ax[0].plot(hist.epoch, hist.history["loss"], label="Train loss")
ax[0].plot(hist.epoch, hist.history["val_loss"], label="Validation loss")
ax[1].set_title('acc')
ax[1].plot(hist.epoch, hist.history["f1"], label="Train F1")
ax[1].plot(hist.epoch, hist.history["val_f1"], label="Validation F1")
ax[0].legend()
ax[1].legend()
plt.savefig('base_model_Adamlr1e-3.png')

bestModel = load_model('./base.model', custom_objects={'f1': f1,'weighted_binary_crossentropy':weighted_binary_crossentropy}) #, 'f1_loss': f1_loss})
# bestModel = load_model('./best.model', custom_objects={'f1': f1,'weighted_binary_crossentropy':weighted_binary_crossentropy}) #, 'f1_loss': f1_loss})

#bestModel = model

fullValGen = vg
lastFullValPred = np.empty((0, 28))
lastFullValLabels = np.empty((0, 28))
for i in tqdm(range(len(fullValGen))):
    im, lbl = fullValGen[i]
    scores = bestModel.predict(im)
    lastFullValPred = np.append(lastFullValPred, scores, axis=0)
    lastFullValLabels = np.append(lastFullValLabels, lbl, axis=0)
print(lastFullValPred.shape, lastFullValLabels.shape)
from sklearn.metrics import f1_score as off1
rng = np.arange(0, 1, 0.001)
f1s = np.zeros((rng.shape[0], 28))
for j,t in enumerate(tqdm(rng)):
    for i in range(28):
        p = np.array(lastFullValPred[:,i]>t, dtype=np.int8)
        scoref1 = off1(lastFullValLabels[:,i], p, average='binary')
        f1s[j,i] = scoref1

print('Individual F1-scores for each class:')
print(np.max(f1s, axis=0))
print('Macro F1-score CV =', np.mean(np.max(f1s, axis=0)))

plt.plot(rng, f1s)
T = np.empty(28)
for i in range(28):
    T[i] = rng[np.where(f1s[:,i] == np.max(f1s[:,i]))[0][0]]
print('Probability threshold maximizing CV F1-score for each class:')
print(T)

pathsTest, labelsTest = getTestDataset()

testg = ProteinDataGenerator(pathsTest, labelsTest, BATCH_SIZE, SHAPE)
submit = pd.read_csv(DIR + '/sample_submission.csv')
P = np.zeros((pathsTest.shape[0], 28))
for i in tqdm(range(len(testg))):
    images, labels = testg[i]
    score = bestModel.predict(images)
    P[i*BATCH_SIZE:i*BATCH_SIZE+score.shape[0]] = score

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

submit['Predicted'] = np.array(prediction)
submit.to_csv('1channels_cnn.csv', index=False)


