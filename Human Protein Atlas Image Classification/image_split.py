import numpy as np
import cv2
from glob import glob
import zipfile
from tqdm import tqdm

size = 512

dir = '/media/dmitri/ボリューム/kaggle/train_full_sizee/'
out_path=dir+'train/'
for i in range(1,24):
    if i<10:
        path=dir+'split_train_train_0'+str(i)
    else:
        path=dir+'split_train_train_'+str(i)
    for name in tqdm(glob()):
        img = cv2.imread(name,cv2.IMREAD_GRAYSCALE)
        # img = Image.open(zip.read(name))
        print(img.shape)
        if img is None:
            print("%s is None",name)
            continue
        v_size = img.shape[0] // size * size
        h_size = img.shape[1] // size * size
        img = img[:v_size, :h_size]

        v_split = img.shape[0] // size
        h_split = img.shape[1] // size
        out_img = []
        [out_img.extend(np.hsplit(h_img, h_split))for h_img in np.vsplit(img, v_split)]
        for num,split_img in enumerate(out_img):
            cv2.imwrite(out_path+name.split('.tif')[0]+'_'+str(num)+'.tif')