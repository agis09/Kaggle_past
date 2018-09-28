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

ImageFile.LOAD_TRUNCATED_IMAGES = True

GAUSSIAN_NOISE = 0.1
UPSAMPLE_MODE = 'SIMPLE'
# number of validation images to use
VALID_IMG_COUNT = 1000
# maximum number of training images
MAX_TRAIN_IMAGES = 1000000
# BASE_MODEL='VGG16'    # ['VGG16', 'RESNET52', 'InceptionV3', 'Xception', 'DenseNet169', 'DenseNet121']
IMG_SIZE = (768, 768)   # [(224, 224), (384, 384), (512, 512), (640, 640)]
BATCH_SIZE = 12     # [1, 8, 16, 24]
DROPOUT = 0.5
DENSE_COUNT = 128
LEARN_RATE = 1e-4
RGB_FLIP = 1    # should rgb be flipped when rendering images

# montage_rgb = lambda x: np.stack([montage(x[:, :, :, i]) for i in range(x.shape[3])], -1)
ship_dir = './'
# train_image_dir = os.path.join(ship_dir, 'train')
test_image_dir = os.path.join(ship_dir, 'test_test')
ship_model = load_model('full_ship_model_VGG16.h5')

ship_model.compile(optimizer=Adam(lr=LEARN_RATE),
                   loss='binary_crossentropy',
                   metrics=['accuracy'])

ship_model.summary()

test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(test_image_dir,
                                                  target_size=IMG_SIZE,
                                                  color_mode='rgb',
                                                  shuffle=False,
                                                  class_mode='binary',
                                                  batch_size=1)
filenames = test_generator.filenames
nb_samples = len(filenames)
hist = ship_model.predict_generator(test_generator,steps=nb_samples,verbose=1)

# print(hist)

image_list = glob(test_image_dir+'./test/*.jpg')
file_name_list = os.listdir(test_image_dir+'./test')

for i,path in enumerate(image_list):
    if hist[i]>=0.5:
        tmp = Image.open(path)
        tmp.save('./test_test_has_ship./test_has_ship./'+file_name_list[i])
    else:
        tmp = Image.open(path)
        tmp.save('./test_has_noship./'+file_name_list[i])



"""             segmentation            """

def IoU(y_true, y_pred, eps=1e-6):
    if np.max(y_true) == 0.0:
        return IoU(1-y_true, 1-y_pred) ## empty image; calc IoU of zeros
    intersection = K.sum(y_true * y_pred, axis=[1,2,3])
    union = K.sum(y_true, axis=[1,2,3]) + K.sum(y_pred, axis=[1,2,3]) - intersection
    return -K.mean( (intersection + eps) / (union + eps), axis=0)


seg_model = load_model("model_unet_with_resnet50.h5",custom_objects={'IoU':IoU})
# seg_model.summary()
seg_model.compile(optimizer=Adam(1e-3, decay=1e-6), loss=IoU, metrics=['binary_accuracy'])

test_image_dir = os.path.join(ship_dir, 'test_test_has_ship./test_has_ship')
test_paths = np.array(os.listdir(test_image_dir))


def predict(img, path=test_image_dir):
    # print(img)
    c_img = imread(os.path.join(path, img))
    c_img = np.expand_dims(c_img, 0)/255.0
    cur_seg = seg_model.predict(c_img)[0]
    cur_seg = binary_opening(cur_seg>0.5, np.expand_dims(disk(2), -1))
    tmp = cur_seg*255.0
    tmp = np.reshape(tmp,(tmp.shape[0],tmp.shape[1]))
    Image.fromarray(np.uint8(tmp)).save('./result./'+img+'_mask.jpg')
    return cur_seg, c_img

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

def pred_encode(img, **kwargs):
    cur_seg, _ = predict(img)
    cur_rles = rle_encode(cur_seg, **kwargs)
    return [img, cur_rles if len(cur_rles) > 0 else None]


out_pred_rows = []


for c_img_name in tqdm(test_paths): ## only a subset as it takes too long to run
    out_pred_rows += [pred_encode(c_img_name, min_threshold=1.0, max_threshold=0.004)]

# print(out_pred_rows)
sub = pd.DataFrame(out_pred_rows)
sub.columns = ['ImageId', 'EncodedPixels']
sub = sub[sub.EncodedPixels.notnull()]

sub1 = pd.read_csv('./sample_submission.csv')
sub1 = pd.DataFrame(np.setdiff1d(sub1['ImageId'].unique(), sub['ImageId'].unique(), assume_unique=True), columns=['ImageId'])
sub1['EncodedPixels'] = None

sub = pd.concat([sub, sub1])
print(sub.head)
sub.to_csv('submission_unet_cnn.csv', index=False)