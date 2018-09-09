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
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

GAUSSIAN_NOISE = 0.1
UPSAMPLE_MODE = 'SIMPLE'
# number of validation images to use
VALID_IMG_COUNT = 1000
# maximum number of training images
MAX_TRAIN_IMAGES = 1000000
BASE_MODEL='DenseNet169'    # ['VGG16', 'RESNET52', 'InceptionV3', 'Xception', 'DenseNet169', 'DenseNet121']
IMG_SIZE = (768, 768)   # [(224, 224), (384, 384), (512, 512), (640, 640)]
BATCH_SIZE = 16     # [1, 8, 16, 24]
DROPOUT = 0.5
DENSE_COUNT = 128
LEARN_RATE = 1e-4
RGB_FLIP = 1    # should rgb be flipped when rendering images

# montage_rgb = lambda x: np.stack([montage(x[:, :, :, i]) for i in range(x.shape[3])], -1)
ship_dir = './'
train_image_dir = os.path.join(ship_dir, 'train')
test_image_dir = os.path.join(ship_dir, 'test')
import gc; gc.enable() # memory is tight
masks = pd.read_csv('./train_ship_segmentations.csv')

print(masks.shape[0], 'masks found')
print(masks['ImageId'].value_counts().shape[0])
masks['path'] = masks['ImageId'].map(lambda x: os.path.join(train_image_dir, x))
# print(masks.head())

masks['ships'] = masks['EncodedPixels'].map(lambda c_row: 1 if isinstance(c_row, str) else 0)
unique_img_ids = masks.groupby('ImageId').agg({'ships': 'sum'}).reset_index()
unique_img_ids['has_ship'] = unique_img_ids['ships'].map(lambda x: 1.0 if x>0 else 0.0)
unique_img_ids['has_ship_vec'] = unique_img_ids['has_ship'].map(lambda x: [x])
masks.drop(['ships'], axis=1, inplace=True)
train_ids, valid_ids = train_test_split(unique_img_ids,
                                        test_size=0.3,
                                        stratify=unique_img_ids['ships'])
train_df = pd.merge(masks, train_ids)
valid_df = pd.merge(masks, valid_ids)
print(train_df.shape[0], 'training masks')
print(valid_df.shape[0], 'validation masks')

train_df = train_df.sample(min(MAX_TRAIN_IMAGES, train_df.shape[0]))
# limit size of training set (otherwise it takes too long)
# train_df[['ships', 'has_ship']].hist()
# exit()
"""         Augment Data            """


if BASE_MODEL == 'VGG16':
    from keras.applications.vgg16 import VGG16 as PTModel, preprocess_input
elif BASE_MODEL == 'RESNET52':
    from keras.applications.resnet50 import ResNet50 as PTModel, preprocess_input
elif BASE_MODEL == 'InceptionV3':
    from keras.applications.inception_v3 import InceptionV3 as PTModel, preprocess_input
elif BASE_MODEL == 'Xception':
    from keras.applications.xception import Xception as PTModel, preprocess_input
elif BASE_MODEL == 'DenseNet169':
    from keras.applications.densenet import DenseNet169 as PTModel, preprocess_input
elif BASE_MODEL == 'DenseNet121':
    from keras.applications.densenet import DenseNet121 as PTModel, preprocess_input
else:
    raise ValueError('Unknown model: {}'.format(BASE_MODEL))

dg_args = dict(featurewise_center=False,
               samplewise_center=False,
               rotation_range=0,
               width_shift_range=0,
               height_shift_range=0,
               shear_range=0,
               # zoom_range = [0.9, 1.0],
               # brightness_range = [0.5, 1.5],
               horizontal_flip=True,
               vertical_flip=True,
               fill_mode='reflect',
               data_format='channels_last',
               preprocessing_function=preprocess_input)
valid_args = dict(fill_mode='reflect',
                  data_format='channels_last',
                  preprocessing_function=preprocess_input)

core_idg = ImageDataGenerator(**dg_args)
valid_idg = ImageDataGenerator(**valid_args)


def flow_from_dataframe(img_data_gen, in_df, path_col, y_col, **dflow_args):
    base_dir = os.path.dirname(in_df[path_col].values[0])
    print('## Ignore next message from keras, values are replaced anyways')
    df_gen = img_data_gen.flow_from_directory(base_dir, class_mode='sparse', **dflow_args)
    df_gen.filenames = in_df[path_col].values
    df_gen.classes = np.stack(in_df[y_col].values)
    df_gen.samples = in_df.shape[0]
    df_gen.n = in_df.shape[0]
    df_gen._set_index_array()
    df_gen.directory = ''   # since we have the full path
    print('Reinserting dataframe: {} images'.format(in_df.shape[0]))
    return df_gen


train_gen = flow_from_dataframe(core_idg, train_df,
                                path_col='path',
                                y_col='has_ship_vec',
                                target_size=IMG_SIZE,
                                color_mode='rgb',
                                batch_size=BATCH_SIZE)

# used a fixed dataset for evaluating the algorithm
valid_x, valid_y = next(flow_from_dataframe(valid_idg,
                                            valid_df,
                                            path_col='path',
                                            y_col='has_ship_vec',
                                            target_size=IMG_SIZE,
                                            color_mode='rgb',
                                            batch_size=VALID_IMG_COUNT))    # one big batch
# print(valid_x.shape, valid_y.shape)
t_x, t_y = next(train_gen)


"""             Build a Model           """


base_pretrained_model = PTModel(input_shape=t_x.shape[1:], include_top=False, weights='imagenet')
base_pretrained_model.trainable = False

img_in = layers.Input(t_x.shape[1:], name='Image_RGB_In')
img_noise = layers.GaussianNoise(GAUSSIAN_NOISE)(img_in)
pt_features = base_pretrained_model(img_noise)
pt_depth = base_pretrained_model.get_output_shape_at(0)[-1]
bn_features = layers.BatchNormalization()(pt_features)
feature_dropout = layers.SpatialDropout2D(DROPOUT)(bn_features)
gmp_dr = layers.GlobalMaxPooling2D()(feature_dropout)
dr_steps = layers.Dropout(DROPOUT)(layers.Dense(DENSE_COUNT, activation='relu')(gmp_dr))
out_layer = layers.Dense(1, activation='sigmoid')(dr_steps)

ship_model = models.Model(inputs=[img_in], outputs=[out_layer], name='full_model')

ship_model.compile(optimizer=Adam(lr=LEARN_RATE),
                   loss='binary_crossentropy',
                   metrics=['accuracy'])

ship_model.summary()

weight_path="{}_weights.best.hdf5".format('boat_detector')

checkpoint = ModelCheckpoint(weight_path, monitor='val_loss', verbose=1,
                             save_best_only=True, mode='min', save_weights_only = True)

reduceLROnPlat = ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=10, verbose=1, mode='auto', epsilon=0.0001, cooldown=5, min_lr=0.0001)
early = EarlyStopping(monitor="val_loss", mode="min", patience=10)
# probably needs to be more patient, but kaggle time is limited
callbacks_list = [checkpoint,reduceLROnPlat]               # [checkpoint, early, reduceLROnPlat]

train_gen.batch_size = BATCH_SIZE
hist = ship_model.fit_generator(train_gen,
                               validation_data=(valid_x, valid_y),
                               epochs=10,
                               callbacks=callbacks_list,
                               workers=3)


"""         result_plot            """


fig, (axL, axR) = plt.subplots(ncols=2, figsize=(10,4))

# loss
def plot_history_loss(hist):
    # Plot the loss in the history
    axL.plot(hist.history['loss'],label="loss for training")
    axL.plot(hist.history['val_loss'],label="loss for validation")
    axL.set_title('model loss')
    axL.set_xlabel('epoch')
    axL.set_ylabel('loss')
    axL.legend(loc='upper right')

# acc
def plot_history_acc(hist):
    # Plot the loss in the history
    axR.plot(hist.history['acc'],label="acc for training")
    axR.plot(hist.history['val_acc'],label="acc for validation")
    axR.set_title('model accuracy')
    axR.set_xlabel('epoch')
    axR.set_ylabel('accuracy')
    axR.legend(loc='lower right')


plot_history_loss(hist)
plot_history_acc(hist)
fig.savefig('./classification_result.png')
plt.close()

ship_model.load_weights(weight_path)
ship_model.save('full_ship_model.h5')

