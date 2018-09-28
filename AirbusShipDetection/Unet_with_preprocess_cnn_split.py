import os
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from skimage.io import imread
import matplotlib.pyplot as plt
from skimage.segmentation import mark_boundaries
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import ImageDataGenerator
# from skimage.util.montage import montage2d as montage
from keras import models, layers
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau
from PIL import ImageFile,Image
from glob import glob
from tqdm import tqdm
from keras.models import load_model
from PIL import Image
import keras.backend as K
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
from glob import glob
ImageFile.LOAD_TRUNCATED_IMAGES = True


"""         load_test_image_split           """


ship_dir = 'F:\\Shiga\\kaggle\\AirbusShipDetection'
# train_image_dir = os.path.join(ship_dir, 'train')
test_image_dir = os.path.join(ship_dir, 'test')

splited_size = 256  # size*size

labels = pd.read_csv('./classification_labels.csv')

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
        img[lo:hi] = 255
    return img.reshape(shape).T  # Needed to align to RLE direction


def masks_as_image(in_mask_list):
    # Take the individual ship masks and create a single mask array for all ships
    all_masks = np.zeros((768, 768), dtype = np.uint8)
    for mask in in_mask_list:
        if isinstance(mask, str):
            all_masks |= rle_decode(mask)
    return all_masks


def image_split(img, size):
    v_split = img.shape[0] // size
    h_split = img.shape[1] // size
    out_img = []
    [out_img.extend(np.hsplit(h_img, h_split)) for h_img in np.vsplit(img, v_split)]
    return out_img


all_batches = os.listdir(test_image_dir)
ids_flag = []


"""                     -----------                         """

ship_model = load_model('full_ship_model_VGG16.h5')

ship_model.compile(optimizer=Adam(lr=1e-4),
                   loss='binary_crossentropy',
                   metrics=['accuracy'])

ship_model.summary()

"""             segmentation_model              """


def IoU(y_true, y_pred, eps=1e-6):
    if np.max(y_true) == 0.0:
        return IoU(1-y_true, 1-y_pred) ## empty image; calc IoU of zeros
    intersection = K.sum(y_true * y_pred, axis=[1,2,3])
    union = K.sum(y_true, axis=[1,2,3]) + K.sum(y_pred, axis=[1,2,3]) - intersection
    return -K.mean( (intersection + eps) / (union + eps), axis=0)


seg_model = load_model("model_unet_with_vgg.h5",custom_objects={'IoU':IoU})
# seg_model.summary()
seg_model.compile(optimizer=Adam(1e-3, decay=1e-6), loss=IoU, metrics=['binary_accuracy'])


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


def pred_encode(img, cur_seg, **kwargs):
    cur_rles = rle_encode(cur_seg, **kwargs)
    return [img, cur_rles if len(cur_rles) > 0 else None]


out_pred_rows = []


for c_img_id in tqdm(all_batches):
    rgb_path = os.path.join(test_image_dir, c_img_id)
    c_img = imread(rgb_path)
    c_img_split = image_split(c_img, splited_size)
    name, ext = os.path.splitext(c_img_id)
    out = []
    for i, img in enumerate(c_img_split):
        image_name = name + '_' + str(i) + ext
        train_img = img / 255.0
        train_img = np.reshape(train_img,(1,train_img.shape[0],train_img.shape[1],3))
        # train_img = Image.fromarray(np.expand_dims(img, 0)/255.0)
        mask_label = ship_model.predict(train_img)

        if mask_label < 0.5:
            mask_img = np.zeros(splited_size*splited_size, dtype=np.uint8).reshape((splited_size,splited_size)).T
        else:
            mask_img = seg_model.predict(train_img)
        # Image.fromarray(mask_img*255).save('./result./' + name + '_mask_' + str(i) + ext)
        mask_img = np.reshape(mask_img,(splited_size,splited_size))
        out.append(Image.fromarray(mask_img))
    result_img = np.vstack((
        np.hstack(out[0:3]),
        np.hstack(out[3:6]),
        np.hstack(out[6:9]),
    ))
    result_img = np.reshape(result_img,(result_img.shape[0],result_img.shape[1],1))
    result_img = binary_opening(result_img >0.5, np.expand_dims(disk(2), -1))

    out_pred_rows += [pred_encode(c_img_id, result_img, min_threshold=1.0, max_threshold=0.004)]

    result_img = result_img*255.0
    result_img = np.reshape(result_img,(768,768)).astype(np.uint8)
    result_img = Image.fromarray(result_img)
    result_img.save('./result./' + name + '_mask_.png')

sub = pd.DataFrame(out_pred_rows)
sub.columns = ['ImageId', 'EncodedPixels']
sub = sub[sub.EncodedPixels.notnull()]

sub1 = pd.read_csv('./sample_submission.csv')
sub1 = pd.DataFrame(np.setdiff1d(sub1['ImageId'].unique(), sub['ImageId'].unique(), assume_unique=True), columns=['ImageId'])
sub1['EncodedPixels'] = None

sub = pd.concat([sub, sub1])
print(sub.head)
sub.to_csv('submission_unet_cnn_split.csv', index=False)