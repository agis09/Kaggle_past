# from https://github.com/killthekitten/kaggle-carvana-2017/blob/master/models.py

from keras.applications.vgg16 import VGG16
from keras.engine.topology import Input
from keras.engine.training import Model
from keras.layers.convolutional import Conv2D, UpSampling2D, Conv2DTranspose
from keras.layers.core import Activation, SpatialDropout2D
from keras.layers.merge import concatenate
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D

from inception_resnet_v2 import InceptionResNetV2
from mobile_net_fixed import MobileNet
from resnet50_fixed import ResNet50
# from param import args

import Unet_with_fine_tuning_models
import losses
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

# montage_rgb = lambda x: np.stack([montage(x[:, :, :, i]) for i in range(x.shape[3])], -1)
ship_dir = 'F:\\Shiga\\kaggle\\AirbusShipDetection'
train_image_dir = os.path.join(ship_dir, 'train_split')
train_mask_image_dir = os.path.join(ship_dir, 'train_mask_split')
input_shape = (256,256,3)


train_df = pd.read_csv("classification_labels.csv")
train_df = train_df[train_df['has_ship']==1]

train_df, valid_df = train_test_split(train_df,test_size = 0.2)



"""         Decode RLEs into Images         """


def make_image_gen(in_df, batch_size):
    all_batches = list(in_df.groupby('ids'))
    out_rgb = []
    out_mask = []
    while True:
        np.random.shuffle(all_batches)
        for c_img_id, c_masks in all_batches:
            c_mask_img_id = c_img_id[:-6]+'_mask'+c_img_id[-6:]
            rgb_path = os.path.join(train_image_dir, c_img_id)
            mask_path = os.path.join(train_mask_image_dir,c_mask_img_id)
            c_img = imread(rgb_path)
            c_mask = imread(mask_path)
            c_mask = np.reshape(c_mask,(c_mask.shape[0],c_mask.shape[1],1))
            out_rgb += [c_img]
            out_mask += [c_mask]
            if len(out_rgb)>=batch_size:
                yield np.stack(out_rgb, 0)/255.0, np.stack(out_mask, 0)/255.0
                out_rgb, out_mask=[], []

"""         Augmentation            """


dg_args = dict(featurewise_center = False,
               samplewise_center = False,
               rotation_range = 45,
               width_shift_range = 0.1,
               height_shift_range = 0.1,
               shear_range = 0.01,
               zoom_range = [0.9, 1.1],
               horizontal_flip = True,
               vertical_flip = True,
               fill_mode = 'reflect',
               data_format = 'channels_last')

image_gen = ImageDataGenerator(**dg_args)
label_gen = ImageDataGenerator(**dg_args)


def create_aug_gen(in_gen, seed = None):
    np.random.seed(seed if seed is not None else np.random.choice(range(9999)))
    for in_x, in_y in in_gen:
        seed = np.random.choice(range(9999))
        # keep the seeds syncronized otherwise the augmentation to the images is different from the masks
        g_x = image_gen.flow(255*in_x,
                             batch_size = in_x.shape[0],
                             seed = seed,
                             shuffle=True)
        g_y = label_gen.flow(in_y,
                             batch_size = in_x.shape[0],
                             seed = seed,
                             shuffle=True)

        yield next(g_x)/255.0, next(g_y)


# t_x, t_y = next(create_aug_gen(train_gen))
gc.collect()

"""         Build a Model           """


def Unet(GAUSSIAN_NOISE=0.1, UPSAMPLE_MODE='SIMPLE', NET_SCALING = (1, 1), EDGE_CROP = 16):

    def upsample_conv(filters, kernel_size, strides, padding):
        return layers.Conv2DTranspose(filters, kernel_size, strides=strides, padding=padding)

    def upsample_simple(filters, kernel_size, strides, padding):
        return layers.UpSampling2D(strides)

    if UPSAMPLE_MODE == 'DECONV':
        upsample = upsample_conv
    else:
        upsample = upsample_simple

    input_img = layers.Input(input_shape, name='RGB_Input')
    pp_in_layer = input_img

    if NET_SCALING is not None:
        pp_in_layer = layers.AvgPool2D(NET_SCALING)(pp_in_layer)

    pp_in_layer = layers.GaussianNoise(GAUSSIAN_NOISE)(pp_in_layer)
    pp_in_layer = layers.BatchNormalization()(pp_in_layer)

    c1 = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(pp_in_layer)
    c1 = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(c1)
    p1 = layers.MaxPooling2D((2, 2))(c1)

    c2 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(p1)
    c2 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(c2)
    p2 = layers.MaxPooling2D((2, 2))(c2)

    c3 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(p2)
    c3 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c3)
    p3 = layers.MaxPooling2D((2, 2))(c3)

    c4 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(p3)
    c4 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c4)
    p4 = layers.MaxPooling2D(pool_size=(2, 2))(c4)

    c5 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(p4)
    c5 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(c5)

    u6 = upsample(128, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = layers.concatenate([u6, c4])
    c6 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(u6)
    c6 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c6)

    u7 = upsample(64, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = layers.concatenate([u7, c3])
    c7 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(u7)
    c7 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c7)

    u8 = upsample(32, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = layers.concatenate([u8, c2])
    c8 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(u8)
    c8 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(c8)

    u9 = upsample(16, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = layers.concatenate([u9, c1], axis=3)
    c9 = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(u9)
    c9 = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(c9)

    d = layers.Conv2D(1, (1, 1), activation='sigmoid')(c9)
    # d = layers.Cropping2D((EDGE_CROP, EDGE_CROP))(d)
    # d = layers.ZeroPadding2D((EDGE_CROP, EDGE_CROP))(d)
    if NET_SCALING is not None:
        d = layers.UpSampling2D(NET_SCALING)(d)

    seg_model = models.Model(inputs=[input_img], outputs=[d])
    seg_model.summary()
    return seg_model


def IoU(y_true, y_pred, eps=1e-6):
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


make_model = Unet_with_fine_tuning_models
model_name = 'vgg'     # resnet50, inception_resnet_v2, mobilenet, vgg, simple_unet
model = make_model.chose_model(input_shape,model_name)

make_loss = losses
loss_name = 'dice_coef_loss'    # IoU, dice_coef_loss, tversky_loss
model.compile(optimizer=Adam(1e-3, decay=1e-6), loss=make_loss.chose_losses(loss_name), metrics=['binary_accuracy'])

weight_path="{}_weights.best.hdf5".format('seg_model')

checkpoint = ModelCheckpoint(weight_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min',
                             save_weights_only=True)

reduceLROnPlat = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                   patience=1, verbose=1, mode='min',
                                   epsilon=0.0001, cooldown=2, min_lr=1e-7)

early = EarlyStopping(monitor="val_loss", mode="min", verbose=2,
                      patience=20) # probably needs to be more patient, but kaggle time is limited

callbacks_list = [checkpoint, early, reduceLROnPlat]
"""
callbacks_list = [checkpoint, reduceLROnPlat]
"""


valid_x, valid_y = next(make_image_gen(valid_df,batch_size=len(valid_df)))

BATCH_SIZE = 16  # resnet50:16    inception_resnet_v2:1   vgg:

# maximum number of steps_per_epoch in training
MAX_TRAIN_STEPS = 7
MAX_TRAIN_EPOCHS = 99

epoch = min(MAX_TRAIN_STEPS, train_df.shape[0]//BATCH_SIZE)
aug_gen = create_aug_gen(make_image_gen(train_df,BATCH_SIZE))
loss_history = [model.fit_generator(aug_gen,
                                    steps_per_epoch=epoch,
                                    epochs=MAX_TRAIN_EPOCHS,
                                    validation_data=(valid_x, valid_y),
                                    callbacks=callbacks_list,
                                    # workers=1 # the generator is not very thread safe
                                   )]


def save_loss(loss_history):
    epich = np.cumsum(np.concatenate(
        [np.linspace(0.5, 1, len(mh.epoch)) for mh in loss_history]))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(22, 10))
    _ = ax1.plot(epich,
                 np.concatenate([mh.history['loss'] for mh in loss_history]),
                 'b-',
                 epich, np.concatenate(
            [mh.history['val_loss'] for mh in loss_history]), 'r-')
    ax1.legend(['Training', 'Validation'])
    ax1.set_title('Loss')

    _ = ax2.plot(epich, np.concatenate(
        [mh.history['binary_accuracy'] for mh in loss_history]), 'b-',
                 epich, np.concatenate(
            [mh.history['val_binary_accuracy'] for mh in loss_history]),
                 'r-')
    ax2.legend(['Training', 'Validation'])
    ax2.set_title('Binary Accuracy (%)')

    fig.savefig('result.png')


save_loss(loss_history)

model.load_weights(weight_path)
model.save('model_unet_with_'+model_name+'.h5')

