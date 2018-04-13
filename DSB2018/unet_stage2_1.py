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
from skimage.io import imread, imshow, imread_collection, concatenate_images,imsave
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
from keras.optimizers import Adam
import tensorflow as tf
from keras.models import Model
from keras.layers import Input
from keras.layers.convolutional import Conv2D, ZeroPadding2D, Conv2DTranspose
from keras.layers.merge import concatenate
from keras.layers import LeakyReLU, BatchNormalization, Activation, Dropout
from skimage.measure import regionprops

IMG_WIDTH = 256
IMG_HEIGHT = 256
IMG_CHANNELS = 3

IMG_SIZE = 256

TRAIN_PATH = 'stage1_tmp/'
TEST_PATH = 'stage2_test_final/'
# TEST_PATH = 'stage1_test/'

seed = 42
random.seed = seed
np.random.seed = seed

train_ids = next(os.walk(TRAIN_PATH))[1]
test_ids = next(os.walk(TEST_PATH))[1]

X_train = np.zeros((len(train_ids)*5, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.float32)
Y_train = np.zeros((len(train_ids)*5, IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.float32)

# 値を-1から1に正規化する関数
def normalize_x(image):
    image = image/127.5 - 1
    return image


# 値を0から1に正規化する関数
def normalize_y(image):
    # image = image/255
    image = np.where(image < 127.5, 0.0, 1.0)
    return image


# 値を0から255に戻す関数
def denormalize_y(image):
    image = image*255
    return image


def random_crop(image1,image2, crop_size=(256, 256)):
    h, w, _ = image1.shape

    # 0~(400-256)の間で画像のtop, leftを決める
    top = np.random.randint(0, h - crop_size[0])
    left = np.random.randint(0, w - crop_size[1])

    # top, leftから画像のサイズである256を足して、bottomとrightを決める
    bottom = top + crop_size[0]
    right = left + crop_size[1]

    # 決めたtop, bottom, left, rightを使って画像を抜き出す
    img1 = image1[top:bottom, left:right, :]
    img2 = image2[top:bottom,left:right,:]
    return img1,img2


def scale_augmentation_crop(image1,image2, scale_range=(300, 512), crop_size=256):
    scale_size = np.random.randint(*scale_range)
    image1 = resize(image1, (scale_size, scale_size), mode='constant', preserve_range=True)
    image2 = resize(image2, (scale_size, scale_size), mode='constant', preserve_range=True)
    img1,img2 = random_crop(image1,image2, (crop_size, crop_size))
    return img1,img2


def LineDrawingDetection(img):
    kernel = np.ones((5,5), np.uint8)
    dilation = cv2.dilate(img, kernel, iterations=1)
    diff = cv2.subtract(img,dilation)
    diff_inv = 255 - diff
    diff_inv_binarized = cv2.threshold(diff_inv,100,255,cv2.THRESH_BINARY)
    return diff_inv


print('Getting and resizing train images and masks ... ')
sys.stdout.flush()

for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):
    path = TRAIN_PATH + id_
    img = imread(path + '/images/' + id_ + '.png')[:,:,:IMG_CHANNELS]
    img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)


    # cv2.imwrite("train_image/train"+str(n)+"_x.bmp",img)

    img = normalize_x(img)
    mask = np.zeros((IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.bool)
    for mask_file in next(os.walk(path + '/masks/'))[2]:
        mask_ = imread(path + '/masks/' + mask_file)
        mask_ = np.expand_dims(resize(mask_, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True), axis=-1)
        mask = np.maximum(mask, mask_)


    # cv2.imwrite("train_image/train"+str(n)+"_y.bmp",mask)


    mask=normalize_y(mask)


    (X_train[n+len(train_ids)*0], Y_train[n+len(train_ids)*0]) = scale_augmentation_crop(img,mask)
    img = np.rot90(img)
    mask = np.rot90(mask)
    (X_train[n + len(train_ids) * 1], Y_train[n + len(train_ids) * 1]) = scale_augmentation_crop(img, mask)
    img = np.rot90(img)
    mask = np.rot90(mask)
    (X_train[n + len(train_ids) * 2], Y_train[n + len(train_ids) * 2]) = scale_augmentation_crop(img, mask)
    img = np.rot90(img)
    mask = np.rot90(mask)
    (X_train[n + len(train_ids) * 3], Y_train[n + len(train_ids) * 3]) = scale_augmentation_crop(img, mask)
    img = img[:,::-1,:]
    mask = mask[:,::-1,:]
    (X_train[n + len(train_ids) * 4], Y_train[n + len(train_ids) * 4]) = scale_augmentation_crop(img, mask)


PATH = 'self_dataset/resize_image/'
train_ids = next(os.walk(PATH))[2]


self_X_train = np.zeros((len(train_ids)*5, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.float32)
self_Y_train = np.zeros((len(train_ids)*5, IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.float32)
file_name1=PATH
file_name2=PATH

for n, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):
    if id_.find('Binary')==-1:
        file_name1=PATH+id_
        continue
    file_name2 = PATH+id_
    img1 = imread(file_name1)
    img2 = imread(file_name2)
    img2=np.reshape(img2,(img2.shape[0],img2.shape[1],1))

    img = normalize_x(img1)
    mask = normalize_y(img2)
    (self_X_train[n + len(train_ids) * 0], self_Y_train[n + len(train_ids) * 0]) = scale_augmentation_crop(img, mask)
    img = np.rot90(img)
    mask = np.rot90(mask)
    (self_X_train[n + len(train_ids) * 1], self_Y_train[n + len(train_ids) * 1]) = scale_augmentation_crop(img, mask)
    img = np.rot90(img)
    mask = np.rot90(mask)
    (self_X_train[n + len(train_ids) * 2], self_Y_train[n + len(train_ids) * 2]) = scale_augmentation_crop(img, mask)
    img = np.rot90(img)
    mask = np.rot90(mask)
    (self_X_train[n + len(train_ids) * 3], self_Y_train[n + len(train_ids) * 3]) = scale_augmentation_crop(img, mask)
    img = img[:, ::-1, :]
    mask = mask[:, ::-1, :]
    (self_X_train[n + len(train_ids) * 4], self_Y_train[n + len(train_ids) * 4]) = scale_augmentation_crop(img, mask)

np.save("X_train",X_train)
np.save("Y_train",Y_train)
np.save("self_X_train",self_X_train)
np.save("self_Y_train",self_Y_train)


X_test = np.zeros((len(test_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.float32)
sizes_test = []
print('Getting and resizing test images ... ')
sys.stdout.flush()
for n, id_ in tqdm(enumerate(test_ids), total=len(test_ids)):
    path = TEST_PATH + id_
    # print(path)
    img = imread(path + '/images/' + id_ + '.png') # [:,:,:IMG_CHANNELS]
    if img.ndim<3:
        img=cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
    else:
        img = imread(path + '/images/' + id_ + '.png')[:,:,:IMG_CHANNELS]

    sizes_test.append([img.shape[0], img.shape[1]])

    cv2.imwrite("test_image/test" + str(n) + "_x.bmp", img)

    img = resize(img, (IMG_HEIGHT, IMG_WIDTH), mode='constant', preserve_range=True)

    img = normalize_x(img)

    X_test[n] = img
np.save("X_test",X_test)


print('Done!')


X_train=np.load("X_train.npy")
Y_train=np.load("Y_train.npy")
X_test=np.load("X_test.npy")
self_X_train=np.load("self_X_train.npy")
self_Y_train=np.load("self_Y_train.npy")
X_train=np.vstack((X_train,self_X_train))
Y_train=np.vstack((Y_train,self_Y_train))



def process(img_rgb):
    #green channel happends to produce slightly better results
    #than the grayscale image and other channels
    img_gray=img_rgb[:,:,1]#cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
    #morphological opening (size tuned on training data)
    circle7=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7))
    img_open=cv2.morphologyEx(img_gray, cv2.MORPH_OPEN, circle7)
    #Otsu thresholding
    img_th=cv2.threshold(img_open,0,255,cv2.THRESH_OTSU)[1]
    #Invert the image in case the objects of interest are in the dark side
    if(np.sum(img_th==255)>np.sum(img_th==0)):
        img_th=cv2.bitwise_not(img_th)
    #second morphological opening (on binary image this time)
    bin_open=cv2.morphologyEx(img_th, cv2.MORPH_OPEN, circle7)
    #connected components
    cc=cv2.connectedComponents(bin_open)[1]
    #cc=segment_on_dt(bin_open,20)
    return cc


def split_and_relabel(mask):
    masks = []
    for i in np.unique(mask):
        if i == 0:  # id = 0 is for background
            continue
        mask_i = (mask == i).astype(np.uint8)
        props = regionprops(mask_i, cache=False)
        if len(props) > 0:
            prop = props[0]
            if prop.convex_area / prop.filled_area > 1.1:
                mask_i = split_mask_v1(mask_i)
        masks.append(mask_i)

    masks = np.array(masks)
    masks_combined = np.amax(masks, axis=0)
    labels = label(masks_combined)
    return labels


def split_mask_v1(mask):
    thresh = mask.copy().astype(np.uint8)
    im2, contours, hierarchy = cv2.findContours(thresh, 2, 1)
    i = 0
    for contour in contours:
        if  cv2.contourArea(contour) > 20:
            hull = cv2.convexHull(contour, returnPoints = False)
            defects = cv2.convexityDefects(contour, hull)
            if defects is None:
                continue
            points = []
            dd = []

            #
            # In this loop we gather all defect points
            # so that they can be filtered later on.
            for i in range(defects.shape[0]):
                s,e,f,d = defects[i,0]
                start = tuple(contour[s][0])
                end = tuple(contour[e][0])
                far = tuple(contour[f][0])
                d = d / 256
                dd.append(d)

            for i in range(len(dd)):
                s,e,f,d = defects[i,0]
                start = tuple(contour[s][0])
                end = tuple(contour[e][0])
                far = tuple(contour[f][0])
                if dd[i] > 1.0 and dd[i]/np.max(dd) > 0.2:
                    points.append(f)

            i = i + 1
            if len(points) >= 2:
                for i in range(len(points)):
                    f1 = points[i]
                    p1 = tuple(contour[f1][0])
                    nearest = None
                    min_dist = np.inf
                    for j in range(len(points)):
                        if i != j:
                            f2 = points[j]
                            p2 = tuple(contour[f2][0])
                            dist = (p1[0]-p2[0])*(p1[0]-p2[0]) + (p1[1]-p2[1])*(p1[1]-p2[1])
                            if dist < min_dist:
                                min_dist = dist
                                nearest = p2

                    cv2.line(thresh,p1, nearest, [0, 0, 0], 2)
    return thresh



def watershed(img,origin):

    thresh, bin_img = cv2.threshold(img,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

    kernel = np.ones((3,3), np.uint8)
    opening = cv2.morphologyEx(bin_img,cv2.MORPH_OPEN,kernel,iterations=2)

    sure_bg = cv2.dilate(opening, kernel, iterations=2)

    dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
    ret,sure_fg = cv2.threshold(dist_transform,0.5*dist_transform.max(),255,0)

    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)

    ret, markers = cv2.connectedComponents(sure_fg)

    # print(np.unique(markers,return_counts=True))
    markers = markers + 1

    markers[unknown == 255] = 0
    markers = cv2.watershed(origin,markers)
    # markers = cv2.watershed(cv2.cvtColor(img,cv2.COLOR_GRAY2BGR), markers)
    # print(np.unique(markers, return_counts=True))
    origin[markers == -1]=[0,255,0]
    # img[markers == -1] = 255
    # img[markers!=1] = 0
    # img = np.reshape(img,(img.shape[0],img.shape[1],1))
    return origin
    # return img


X_train_tmp = np.zeros((len(X_train), IMG_HEIGHT, IMG_WIDTH, 3), dtype=np.float32)
X_test_tmp = np.zeros((len(test_ids), IMG_HEIGHT, IMG_WIDTH, 3), dtype=np.float32)
for i,x in enumerate(X_train):
    x = (x+1)*127.5
    x = x.astype(np.uint8)
    origin = x
    x = cv2.cvtColor(x, cv2.COLOR_RGB2GRAY)
    X_train_tmp[i]=normalize_x(watershed(x,origin))
for i,x in enumerate(X_test):
    x = (x+1)*127.5
    x = x.astype(np.uint8)
    origin = x
    x = cv2.cvtColor(x, cv2.COLOR_RGB2GRAY)
    X_test_tmp[i]=normalize_x(watershed(x,origin))
X_train = X_train_tmp
X_test=X_test_tmp
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


def dice_coef(y_true, y_pred):
    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)
    intersection = K.sum(y_true * y_pred)
    return 2.0 * intersection / (K.sum(y_true) + K.sum(y_pred) + 1)


def dice_coef_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)

"""

class UNet(object):
    def __init__(self, input_channel_count, output_channel_count, first_layer_filter_count):
        self.INPUT_IMAGE_SIZE = 256
        self.CONCATENATE_AXIS = -1
        self.CONV_FILTER_SIZE = 4
        self.CONV_STRIDE = 2
        self.CONV_PADDING = (1, 1)
        self.DECONV_FILTER_SIZE = 2
        self.DECONV_STRIDE = 2

        # (256 x 256 x input_channel_count)
        inputs = Input((self.INPUT_IMAGE_SIZE, self.INPUT_IMAGE_SIZE, input_channel_count))

        # エンコーダーの作成
        # (128 x 128 x N)
        enc1 = ZeroPadding2D(self.CONV_PADDING)(inputs)
        enc1 = Conv2D(first_layer_filter_count, self.CONV_FILTER_SIZE, strides=self.CONV_STRIDE)(enc1)

        # (64 x 64 x 2N)
        filter_count = first_layer_filter_count*2
        enc2 = self._add_encoding_layer(filter_count, enc1)

        # (32 x 32 x 4N)
        filter_count = first_layer_filter_count*4
        enc3 = self._add_encoding_layer(filter_count, enc2)

        # (16 x 16 x 8N)
        filter_count = first_layer_filter_count*8
        enc4 = self._add_encoding_layer(filter_count, enc3)

        # (8 x 8 x 8N)
        enc5 = self._add_encoding_layer(filter_count, enc4)

        # (4 x 4 x 8N)
        enc6 = self._add_encoding_layer(filter_count, enc5)

        # (2 x 2 x 8N)
        enc7 = self._add_encoding_layer(filter_count, enc6)

        # (1 x 1 x 8N)
        enc8 = self._add_encoding_layer(filter_count, enc7)

        # デコーダーの作成
        # (2 x 2 x 8N)
        dec1 = self._add_decoding_layer(filter_count, True, enc8)
        dec1 = concatenate([dec1, enc7], axis=self.CONCATENATE_AXIS)

        # (4 x 4 x 8N)
        dec2 = self._add_decoding_layer(filter_count, True, dec1)
        dec2 = concatenate([dec2, enc6], axis=self.CONCATENATE_AXIS)

        # (8 x 8 x 8N)
        dec3 = self._add_decoding_layer(filter_count, True, dec2)
        dec3 = concatenate([dec3, enc5], axis=self.CONCATENATE_AXIS)

        # (16 x 16 x 8N)
        dec4 = self._add_decoding_layer(filter_count, False, dec3)
        dec4 = concatenate([dec4, enc4], axis=self.CONCATENATE_AXIS)

        # (32 x 32 x 4N)
        filter_count = first_layer_filter_count*4
        dec5 = self._add_decoding_layer(filter_count, False, dec4)
        dec5 = concatenate([dec5, enc3], axis=self.CONCATENATE_AXIS)

        # (64 x 64 x 2N)
        filter_count = first_layer_filter_count*2
        dec6 = self._add_decoding_layer(filter_count, False, dec5)
        dec6 = concatenate([dec6, enc2], axis=self.CONCATENATE_AXIS)

        # (128 x 128 x N)
        filter_count = first_layer_filter_count
        dec7 = self._add_decoding_layer(filter_count, False, dec6)
        dec7 = concatenate([dec7, enc1], axis=self.CONCATENATE_AXIS)

        # (256 x 256 x output_channel_count)
        dec8 = Activation(activation='relu')(dec7)
        dec8 = Conv2DTranspose(output_channel_count, self.DECONV_FILTER_SIZE, strides=self.DECONV_STRIDE)(dec8)
        dec8 = Activation(activation='sigmoid')(dec8)

        self.UNET = Model(input=inputs, output=dec8)

    def _add_encoding_layer(self, filter_count, sequence):
        new_sequence = LeakyReLU(0.2)(sequence)
        new_sequence = ZeroPadding2D(self.CONV_PADDING)(new_sequence)
        new_sequence = Conv2D(filter_count, self.CONV_FILTER_SIZE, strides=self.CONV_STRIDE)(new_sequence)
        new_sequence = BatchNormalization()(new_sequence)
        return new_sequence

    def _add_decoding_layer(self, filter_count, add_drop_layer, sequence):
        new_sequence = Activation(activation='relu')(sequence)
        new_sequence = Conv2DTranspose(filter_count, self.DECONV_FILTER_SIZE, strides=self.DECONV_STRIDE,
                                       kernel_initializer='he_uniform')(new_sequence)
        new_sequence = BatchNormalization()(new_sequence)
        if add_drop_layer:
            new_sequence = Dropout(0.5)(new_sequence)
        return new_sequence

    def get_model(self):
        return self.UNET


# 入力はBGR3チャンネル
input_channel_count = 3
# 出力はグレースケール1チャンネル
output_channel_count = 1
# 一番初めのConvolutionフィルタ枚数は64
first_layer_filter_count = 64

network = UNet(input_channel_count, output_channel_count, first_layer_filter_count)
model = network.get_model()
"""

inputs = Input((IMG_HEIGHT, IMG_WIDTH, 3))

c1 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (inputs)
c1 = Dropout(0.1) (c1)
c1 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c1)
p1 = MaxPooling2D((2, 2)) (c1)

c2 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p1)
c2 = Dropout(0.1) (c2)
c2 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c2)
p2 = MaxPooling2D((2, 2)) (c2)

c3 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p2)
c3 = Dropout(0.2) (c3)
c3 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c3)
p3 = MaxPooling2D((2, 2)) (c3)

c4 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p3)
c4 = Dropout(0.2) (c4)
c4 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c4)
p4 = MaxPooling2D(pool_size=(2, 2)) (c4)

c5 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (p4)
c5 = Dropout(0.3) (c5)
c5 = Conv2D(256, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c5)

u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same') (c5)
u6 = concatenate([u6, c4])
c6 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u6)
c6 = Dropout(0.2) (c6)
c6 = Conv2D(128, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c6)

u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same') (c6)
u7 = concatenate([u7, c3])
c7 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u7)
c7 = Dropout(0.2) (c7)
c7 = Conv2D(64, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c7)

u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same') (c7)
u8 = concatenate([u8, c2])
c8 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u8)
c8 = Dropout(0.1) (c8)
c8 = Conv2D(32, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c8)

u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same') (c8)
u9 = concatenate([u9, c1], axis=3)
c9 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (u9)
c9 = Dropout(0.1) (c9)
c9 = Conv2D(16, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same') (c9)

outputs = Conv2D(1, (1, 1), activation='sigmoid') (c9)

model = Model(inputs=[inputs], outputs=[outputs])
# model.summary()
# model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=[my_iou_metric])
model.compile(loss='binary_crossentropy', optimizer=Adam(), metrics=[dice_coef])


BATCH_SIZE = 12
NUM_EPOCH = 100
checkpointer = ModelCheckpoint('model-dsbowl2018-1.h5', verbose=1, save_best_only=True)
history = model.fit(X_train, Y_train, batch_size=BATCH_SIZE, epochs=NUM_EPOCH, verbose=1,validation_data=(X_val,Y_val),
                    callbacks=[checkpointer])
# model.save_weights('unet_weights.hdf5')


#############  predict  ###############

model = load_model('model-dsbowl2018-1.h5', custom_objects={'dice_coef': dice_coef})
# model.load_weights('unet_weights.hdf5')
preds_test = model.predict(X_test,verbose=1)
preds_test_upsampled = []




for i in range(len(preds_test)):
    tmp = preds_test[i]
    tmp=resize(tmp,(sizes_test[i][0],sizes_test[i][1]), mode='constant', preserve_range=True)
    tmp=denormalize_y(tmp)
    cv2.imwrite("test_image/test"+str(i)+"_y.bmp",tmp)
    # tmp = cv2.imread("test_image/test"+str(i)+"_y.bmp")
    # tmp = split_and_relabel(process(tmp))
    # cv2.imwrite("test_image/test"+str(i)+"_y_split.bmp",tmp)
    tmp=cv2.imread("test_image/test"+str(i)+"_y.bmp", cv2.COLOR_BGR2GRAY)
    # watershed(tmp)
    # cv2.imwrite("test_image/test" + str(i) + "_y_water.bmp", tmp)
    print("saving...")



for i in range(len(preds_test)):
    preds_test_upsampled.append(resize(preds_test[i],
                                       (sizes_test[i][0], sizes_test[i][1]), mode='constant', preserve_range=True))


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

print(len(set(new_test_ids)))
sub = pd.DataFrame()
sub['ImageId'] = new_test_ids
sub['EncodedPixels'] = pd.Series(rles).apply(lambda x: ' '.join(str(y) for y in x))
sub.to_csv('sub-test.csv', index=False)