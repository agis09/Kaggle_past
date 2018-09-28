import numpy as np
import os
import os.path
import pandas as pd
from skimage.io import imread
import matplotlib.pyplot as plt
from skimage.segmentation import mark_boundaries
from skimage.morphology import binary_opening, disk
from PIL import Image
from tqdm import tqdm
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

ship_dir = 'F:\\Shiga\\kaggle\\AirbusShipDetection'
train_image_dir = os.path.join(ship_dir, 'train')
test_image_dir = os.path.join(ship_dir, 'test')

splited_size = 256  # size*size



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


masks = pd.read_csv(os.path.join(ship_dir, 'train_ship_segmentations.csv'))

def image_split(img, size):
    v_split = img.shape[0] // size
    h_split = img.shape[1] // size
    out_img = []
    [out_img.extend(np.hsplit(h_img, h_split)) for h_img in np.vsplit(img, v_split)]
    return out_img


all_batches = list(masks.groupby('ImageId'))
ids_flag = []

for c_img_id, c_masks in tqdm(all_batches):
    # print(c_masks)
    rgb_path = os.path.join(train_image_dir, c_img_id)
    c_img = imread(rgb_path)
    c_mask = np.expand_dims(masks_as_image(c_masks['EncodedPixels'].values), -1)
    c_img_split = image_split(c_img, splited_size)
    c_mask_split = image_split(c_mask, splited_size)
    name, ext = os.path.splitext(c_img_id)
    c_mask = np.reshape(c_mask,(768,768))
    Image.fromarray(c_mask).save('./train_mask./'+name+'_mask_'+ext)
    for i, img in enumerate(c_img_split):
        Image.fromarray(img).save('./train_split./' + name + '_' + str(i) + ext)

    for i, img in enumerate(c_mask_split):
        img = np.reshape(img,(img.shape[0],img.shape[1]))
        Image.fromarray(img).save('./train_mask_split./' + name + '_mask_' + str(i) + ext)
        ids_flag.append((name+'_'+str(i)+ext,int(len(np.where(img!=0)[0])>0)))


df = pd.DataFrame(ids_flag,columns=['ids','has_ship'])
df.to_csv('classification_labels.csv',index=False)