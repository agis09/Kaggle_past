import os
import sys
import random
import warnings

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
from itertools import chain
from skimage.io import imread, imshow, imread_collection, concatenate_images
from skimage import color,exposure
from skimage.transform import resize
from skimage.morphology import label
from sklearn.model_selection import train_test_split
from keras.models import Model, load_model
from keras.layers import Input
from keras.layers.core import Dropout, Lambda
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K

import tensorflow as tf

IMG_WIDTH = 256
IMG_HEIGHT = 256
IMG_CHANNELS = 3

TRAIN_PATH = 'stage1_train/'
TEST_PATH = 'stage1_test/'

seed = 42
random.seed = seed
np.random.seed = seed

train_ids = next(os.walk(TRAIN_PATH))[1]
test_ids = next(os.walk(TEST_PATH))[1]
"""
X_train = np.zeros((len(train_ids)*8, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
Y_train = np.zeros((len(train_ids)*8, IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)
print('Getting and resizing train images and masks ... ')
sys.stdout.flush()

for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):
    path = TRAIN_PATH + id_
    img = imread(path + '/images/' + id_ + '.png')[:,:,:IMG_CHANNELS]
    img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)


    X_train[n] = img
    X_train[n+len(train_ids)]=np.rot90(img)
    X_train[n+len(train_ids)*2]=np.rot90(img,2)
    X_train[n+len(train_ids)*3]=np.rot90(img,3)
    img = cv2.flip(img,1)
    X_train[n+len(train_ids)*4]=img
    X_train[n+len(train_ids)*5]=np.rot90(img)
    X_train[n+len(train_ids)*6]=np.rot90(img,2)
    X_train[n+len(train_ids)*7]=np.rot90(img,3)




    mask = np.zeros((IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)
    for mask_file in next(os.walk(path + '/masks/'))[2]:
        mask_ = imread(path + '/masks/' + mask_file)
        mask_ = np.expand_dims(resize(mask_, (IMG_HEIGHT, IMG_WIDTH), mode='constant',
                                      preserve_range=True), axis=-1)
        mask = np.maximum(mask, mask_)


    Y_train[n] = mask
    Y_train[n+len(train_ids)]=np.rot90(mask)
    Y_train[n+len(train_ids)*2]=np.rot90(mask,2)
    Y_train[n+len(train_ids)*3]=np.rot90(mask,3)
    mask=np.reshape(cv2.flip(mask,1),(IMG_HEIGHT,IMG_WIDTH,1))
    Y_train[n+len(train_ids)*4]=mask
    Y_train[n+len(train_ids)*5]=np.rot90(mask)
    Y_train[n+len(train_ids)*6]=np.rot90(mask,2)
    Y_train[n+len(train_ids)*7]=np.rot90(mask,3)


np.save("X_train",X_train)
np.save("Y_train",Y_train)

"""
X_test = np.zeros((len(test_ids)*8, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.uint8)
sizes_test = []
print('Getting and resizing test images ... ')
sys.stdout.flush()
for n, id_ in tqdm(enumerate(test_ids), total=len(test_ids)):
    path = TEST_PATH + id_
    img = imread(path + '/images/' + id_ + '.png')[:,:,:IMG_CHANNELS]
    sizes_test.append([img.shape[0], img.shape[1]])
    img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)
    X_test[n] = img
    X_test[n+len(test_ids)]=np.rot90(img)
    X_test[n+len(test_ids)*2]=np.rot90(img,2)
    X_test[n+len(test_ids)*3]=np.rot90(img,3)
    img = cv2.flip(img,1)
    X_test[n+len(test_ids)*4]=img
    X_test[n+len(test_ids)*5]=np.rot90(img)
    X_test[n+len(test_ids)*6]=np.rot90(img,2)
    X_test[n+len(test_ids)*7]=np.rot90(img,3)


print('Done!')


X_train=np.load("X_train.npy")
Y_train=np.load("Y_train.npy")


for j, x in enumerate(X_train):
    hsv = color.rgb2hsv(x)
    x = color.hsv2rgb(cv2.equalizeHist(hsv[:,:,2]))


for j, x in enumerate(X_test):
    hsv = color.rgb2hsv(x)
    x = color.hsv2rgb(cv2.equalizeHist(hsv[:,:,2]))


(X_train,X_val,Y_train,Y_val)=train_test_split(X_train,Y_train,test_size=0.1)




# Define IoU metric
def iou_metric(y_true_in, y_pred_in, print_table=False):
    labels = label(y_true_in > 0.5)
    y_pred = label(y_pred_in > 0.5)

    true_objects = len(np.unique(labels))
    pred_objects = len(np.unique(y_pred))

    intersection = np.histogram2d(labels.flatten(), y_pred.flatten(), bins=(true_objects, pred_objects))[0]

    # Compute areas (needed for finding the union between all objects)
    area_true = np.histogram(labels, bins=true_objects)[0]
    area_pred = np.histogram(y_pred, bins=pred_objects)[0]
    area_true = np.expand_dims(area_true, -1)
    area_pred = np.expand_dims(area_pred, 0)

    # Compute union
    union = area_true + area_pred - intersection

    # Exclude background from the analysis
    intersection = intersection[1:, 1:]
    union = union[1:, 1:]
    union[union == 0] = 1e-9

    # Compute the intersection over union
    iou = intersection / union

    # Precision helper function
    def precision_at(threshold, iou):
        matches = iou > threshold
        true_positives = np.sum(matches, axis=1) == 1  # Correct objects
        false_positives = np.sum(matches, axis=0) == 0  # Missed objects
        false_negatives = np.sum(matches, axis=1) == 0  # Extra objects
        tp, fp, fn = np.sum(true_positives), np.sum(false_positives), np.sum(false_negatives)
        return tp, fp, fn

    # Loop over IoU thresholds
    prec = []
    if print_table:
        print("Thresh\tTP\tFP\tFN\tPrec.")
    for t in np.arange(0.5, 1.0, 0.05):
        tp, fp, fn = precision_at(t, iou)
        if (tp + fp + fn) > 0:
            p = tp / (tp + fp + fn)
        else:
            p = 0
        if print_table:
            print("{:1.3f}\t{}\t{}\t{}\t{:1.3f}".format(t, tp, fp, fn, p))
        prec.append(p)

    if print_table:
        print("AP\t-\t-\t-\t{:1.3f}".format(np.mean(prec)))
    return np.mean(prec)


def iou_metric_batch(y_true_in, y_pred_in):
    batch_size = y_true_in.shape[0]
    metric = []
    for batch in range(batch_size):
        value = iou_metric(y_true_in[batch], y_pred_in[batch])
        metric.append(value)
    return np.array(np.mean(metric), dtype=np.float32)


def my_iou_metric(label, pred):
    metric_value = tf.py_func(iou_metric_batch, [label, pred], tf.float32)
    return metric_value

# Build U-Net model
inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
s = Lambda(lambda x: x / 255) (inputs)

c1 = Conv2D(8, (3, 3), activation='relu', padding='same') (s)
c1 = Conv2D(8, (3, 3), activation='relu', padding='same') (c1)
p1 = MaxPooling2D((2, 2)) (c1)

c2 = Conv2D(16, (3, 3), activation='relu', padding='same') (p1)
c2 = Conv2D(16, (3, 3), activation='relu', padding='same') (c2)
p2 = MaxPooling2D((2, 2)) (c2)

c3 = Conv2D(32, (3, 3), activation='relu', padding='same') (p2)
c3 = Conv2D(32, (3, 3), activation='relu', padding='same') (c3)
p3 = MaxPooling2D((2, 2)) (c3)

c4 = Conv2D(64, (3, 3), activation='relu', padding='same') (p3)
c4 = Conv2D(64, (3, 3), activation='relu', padding='same') (c4)
p4 = MaxPooling2D(pool_size=(2, 2)) (c4)

c5 = Conv2D(128, (3, 3), activation='relu', padding='same') (p4)
c5 = Conv2D(128, (3, 3), activation='relu', padding='same') (c5)

u6 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same') (c5)
u6 = concatenate([u6, c4])
c6 = Conv2D(64, (3, 3), activation='relu', padding='same') (u6)
c6 = Conv2D(64, (3, 3), activation='relu', padding='same') (c6)

u7 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same') (c6)
u7 = concatenate([u7, c3])
c7 = Conv2D(32, (3, 3), activation='relu', padding='same') (u7)
c7 = Conv2D(32, (3, 3), activation='relu', padding='same') (c7)

u8 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same') (c7)
u8 = concatenate([u8, c2])
c8 = Conv2D(16, (3, 3), activation='relu', padding='same') (u8)
c8 = Conv2D(16, (3, 3), activation='relu', padding='same') (c8)

u9 = Conv2DTranspose(8, (2, 2), strides=(2, 2), padding='same') (c8)
u9 = concatenate([u9, c1], axis=3)
c9 = Conv2D(8, (3, 3), activation='relu', padding='same') (u9)
c9 = Conv2D(8, (3, 3), activation='relu', padding='same') (c9)

outputs = Conv2D(1, (1, 1), activation='sigmoid') (c9)

model = Model(inputs=[inputs], outputs=[outputs])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[my_iou_metric])
model.summary()

# Fit model
# earlystopper = EarlyStopping(patience=5, verbose=1)
checkpointer = ModelCheckpoint('model-dsbowl2018-1.h5', verbose=1, save_best_only=True)
results = model.fit(X_train, Y_train, validation_split=0.2, batch_size=8, epochs=30,verbose=1)


#############  predict  ###############

model = load_model('model-dsbowl2018-1.h5',custom_objects={'my_iou_metric': my_iou_metric})
preds_test = model.predict(X_test, verbose=1)


for n, id_ in tqdm(enumerate(test_ids), total=len(test_ids)):

    preds_test[n] += np.rot90(preds_test[n+len(test_ids)],3)+np.rot90(preds_test[n+len(test_ids)*2],2) +np.rot90(preds_test[n+len(test_ids)*3],1)

    img1 = cv2.flip(preds_test[n+len(test_ids)*4],1)
    img1=np.reshape(img1,(IMG_HEIGHT,IMG_WIDTH,1))
    img2 = np.rot90(preds_test[n+len(test_ids)*5],3)
    img2 = cv2.flip(img2,1)
    img2=np.reshape(img2,(IMG_HEIGHT,IMG_WIDTH,1))
    img3 = np.rot90(preds_test[n+len(test_ids)*6],2)
    img3 = cv2.flip(img3,1)
    img3=np.reshape(img3,(IMG_HEIGHT,IMG_WIDTH,1))
    img4 = np.rot90(preds_test[n+len(test_ids)*7],1)
    img4 = np.rot90(img4,1)
    img4=np.reshape(img4,(IMG_HEIGHT,IMG_WIDTH,1))
    preds_test[n]+=img1+img2+img3+img4


preds_test = preds_test[:len(test_ids),:,:,:]

preds_test_t = (preds_test > 0.5).astype(np.uint8)
preds_test_upsampled = []
for i in range(len(preds_test)):
    preds_test_upsampled.append(resize(np.squeeze(preds_test[i]),
                                       (sizes_test[i][0], sizes_test[i][1]),
                                       mode='constant', preserve_range=True))


def rle_encoding(x):
    dots = np.where(x.T.flatten() == 1)[0]
    run_lengths = []
    prev = -2
    for b in dots:
        if (b>prev+1): run_lengths.extend((b + 1, 0))
        run_lengths[-1] += 1
        prev = b
    return run_lengths


def prob_to_rles(x, cutoff=0.5):
    lab_img = label(x > cutoff)
    for i in range(1, lab_img.max() + 1):
        yield rle_encoding(lab_img == i)

new_test_ids = []
rles = []
for n, id_ in enumerate(test_ids):
    rle = list(prob_to_rles(preds_test_upsampled[n]))
    rles.extend(rle)
    new_test_ids.extend([id_] * len(rle))

sub = pd.DataFrame()
sub['ImageId'] = new_test_ids
sub['EncodedPixels'] = pd.Series(rles).apply(lambda x: ' '.join(str(y) for y in x))
sub.to_csv('sub-test-aug.csv', index=False)
