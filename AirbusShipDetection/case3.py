import os
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from skimage.io import imread
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
train_image_dir = os.path.join(ship_dir, 'train')
test_image_dir = os.path.join(ship_dir, 'test')


def multi_rle_encode(img):
    labels = label(img)
    if img.ndim > 2:
        return [rle_encode(np.sum(labels==k, axis=2)) for k in np.unique(labels[labels>0])]
    else:
        return [rle_encode(labels==k) for k in np.unique(labels[labels>0])]
# ref: https://www.kaggle.com/paulorzp/run-length-encode-and-decode


def rle_encode(img, min_threshold=1e-3, max_threshold=None):
    '''
    img: numpy array, 1 - mask, 0 - background
    Returns run length as string formated
    '''
    if np.max(img) < min_threshold:
        return '' ## no need to encode if it's all zeros
    if max_threshold and np.mean(img) > max_threshold:
        return '' ## ignore overfilled mask
    pixels = img.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


def rle_decode(mask_rle, shape=(768, 768)):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return
    Returns numpy array, 1 - mask, 0 - background
    '''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape).T  # Needed to align to RLE direction


def masks_as_image(in_mask_list):
    # Take the individual ship masks and create a single mask array for all ships
    all_masks = np.zeros((768, 768), dtype = np.uint8)
    for mask in in_mask_list:
        if isinstance(mask, str):
            all_masks |= rle_decode(mask)
    return all_masks


def masks_as_color(in_mask_list):
    # Take the individual ship masks and create a color mask array for each ships
    all_masks = np.zeros((768, 768), dtype = np.float)
    scale = lambda x: (len(in_mask_list)+x+1) / (len(in_mask_list)*2) ## scale the heatmap image to shift
    for i,mask in enumerate(in_mask_list):
        if isinstance(mask, str):
            all_masks[:,:] += scale(i) * rle_decode(mask)
    return all_masks

"""
masks = pd.read_csv(os.path.join(ship_dir, 'train_ship_segmentations.csv'))
# not_empty = pd.notna(masks.EncodedPixels)
masks['ships'] = masks['EncodedPixels'].map(lambda c_row: 1 if isinstance(c_row, str) else 0)
unique_img_ids = masks.groupby('ImageId').agg({'ships': 'sum'}).reset_index()
unique_img_ids['has_ship'] = unique_img_ids['ships'].map(lambda x: 1.0 if x>0 else 0.0)
         # Undersample Empty Images            
SAMPLES_PER_GROUP = 2000
balanced_train_df = unique_img_ids.groupby('ships').apply(lambda x: x.sample(SAMPLES_PER_GROUP) \
                                                                    if len(x) > SAMPLES_PER_GROUP else x)
print(balanced_train_df.shape[0], 'masks')
train_ids, valid_ids = train_test_split(balanced_train_df,
                                        test_size = 0.2,
                                        stratify = balanced_train_df['ships'])
train_df = pd.merge(masks, train_ids)
valid_df = pd.merge(masks, valid_ids)
train_df.to_csv("train_df.csv")
valid_df.to_csv("valid_df.csv")
"""
train_df=pd.read_csv("train_df.csv")
valid_df=pd.read_csv("valid_df.csv")
print(train_df.shape[0], 'training masks')
print(valid_df.shape[0], 'validation masks')

# print(train_df)

"""         Decode RLEs into Images         """


def make_image_gen(in_df, batch_size):
    all_batches = list(in_df.groupby('ImageId'))
    out_rgb = []
    out_mask = []
    while True:
        np.random.shuffle(all_batches)
        for c_img_id, c_masks in all_batches:
            rgb_path = os.path.join(train_image_dir, c_img_id)
            c_img = imread(rgb_path)
            c_mask = np.expand_dims(masks_as_image(c_masks['EncodedPixels'].values), -1)
            out_rgb += [c_img]
            out_mask += [c_mask]
            if len(out_rgb)>=batch_size:
                yield np.stack(out_rgb, 0)/255.0, np.stack(out_mask, 0)
                out_rgb, out_mask=[], []



# train_gen = make_image_gen(train_df, batch_size=48)
# train_x, train_y = next(train_gen)
# print(train_y.shape)


"""         Augmentation            """


dg_args = dict(featurewise_center = False,
               samplewise_center = False,
               rotation_range = 45,
               width_shift_range = 0.1,
               height_shift_range = 0.1,
               shear_range = 0.01,
               zoom_range = [0.9, 1.25],
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

    input_img = layers.Input((768,768,3), name='RGB_Input')
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


## IoU of boats
def IoU(y_true, y_pred, eps=1e-6):
    if np.max(y_true) == 0.0:
        return IoU(1-y_true, 1-y_pred) ## empty image; calc IoU of zeros
    intersection = K.sum(y_true * y_pred, axis=[1,2,3])
    union = K.sum(y_true, axis=[1,2,3]) + K.sum(y_pred, axis=[1,2,3]) - intersection
    return -K.mean( (intersection + eps) / (union + eps), axis=0)


model=Unet()
model.compile(optimizer=Adam(1e-3, decay=1e-6), loss=IoU, metrics=['binary_accuracy'])

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
VALID_IMG = 600
valid_x, valid_y = next(make_image_gen(valid_df,batch_size=VALID_IMG))

BATCH_SIZE = 8

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
model.save('model.h5')

"""         Submission          """


def predict(img, path=test_image_dir):
    c_img = imread(os.path.join(path, c_img_name))
    c_img = np.expand_dims(c_img, 0)/255.0
    cur_seg = model.predict(c_img)[0]
    cur_seg = binary_opening(cur_seg>0.99, np.expand_dims(disk(2), -1))
    return cur_seg, c_img


test_paths = np.array(os.listdir(test_image_dir))
## Find a threshold to select single ships
ship1_x, ship1_y = next(make_image_gen(valid_df[valid_df['ships'] == 1], VALID_IMG))
max_threshold  = np.mean(ship1_y)
print('Max mean threshold:', max_threshold) ## HACK: ignore imgs with too many trues/boats to avoid submitting masks that should have been split


def pred_encode(img, **kwargs):
    cur_seg, _ = predict(img)
    cur_rles = rle_encode(cur_seg, **kwargs)
    return [img, cur_rles if len(cur_rles) > 0 else None]


out_pred_rows = []


for c_img_name in tqdm(test_paths): ## only a subset as it takes too long to run
    out_pred_rows += [pred_encode(c_img_name, min_threshold=1.0, max_threshold=max_threshold)]

print(out_pred_rows)
sub = pd.DataFrame(out_pred_rows)
sub.columns = ['ImageId', 'EncodedPixels']
sub = sub[sub.EncodedPixels.notnull()]

sub1 = pd.read_csv('./sample_submission.csv')
sub1 = pd.DataFrame(np.setdiff1d(sub1['ImageId'].unique(), sub['ImageId'].unique(), assume_unique=True), columns=['ImageId'])
sub1['EncodedPixels'] = None

sub = pd.concat([sub, sub1])
print(sub.head)
sub.to_csv('submission.csv', index=False)
sub.head()
