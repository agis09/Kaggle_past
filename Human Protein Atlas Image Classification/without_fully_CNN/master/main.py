# from https://www.kaggle.com/rejpalcz/gapnet-pl-lb-0-385
import sys
#import cv2
#import numpy as np
import pandas as pd

from tensorflow import set_random_seed
# from keras.models import Sequential, load_model
# from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau, TensorBoard
# from keras.optimizers import Adam, SGD
# from keras import regularizers, metrics
# from keras import backend as K

#from scipy import ndimage
import signal
from pathlib import Path

import DataSet
from ProteinDataGenerator import ProteinDataGenerator
from f1 import *
from model import *
from pretrain import pretrain_with_lr
from train import train

def handler(signal, frame):
        print('Key-interrupted')
        sys.exit(0)
signal.signal(signal.SIGINT, handler)


BATCH_SIZE = 16
SEED = 777

#set_random_seed(SEED)
SHAPE = (512, 512, 4)
DIR = '../'
imagePath = '/home/elsa/Downloads/train_full_size'
csvPath = '/home/elsa/kaggle/protein/train.csv'
VAL_RATIO = 0.2  # 20 % as validation
THRESHOLD = 0.5  # due to different cost of True Positive vs False Positive, this is the probability threshold to predict the class as 'yes'


POS_WEIGHT = 20  # multiplier for positive targets, needs to be tuned


def pretrain():
    """
    source : ディレクトリtrainとtrain.csvの位置するパス
    csv : csvファイルのパス
    """
    lr_list=[1e-4,1e-5,1e-6,1e-7]
    df = pd.read_csv(csvPath)
    for lr in lr_list:
        print(lr)
        res_dir = Path('/home/elsa/kaggle/protein/kaneko0035_results','lr'+format(lr))
        res_dir.mkdir(parents=True, exist_ok=True)
        pretrain_with_lr(imagePath, df, res_dir, BATCH_SIZE, SHAPE, lr)
    return 1

if __name__ == '__main__':
    flag =True
    if flag:
        pretrain()
    else:
        print("\n\n#####################___________model_label_train______________#########################\n\n")
        df = pd.read_csv(csvPath)
        weight = '/home/elsa/kaggle/protein/kaggle_protein/pretrain_results/lr1e-05/next_base.model0.000010'
        res_dir = Path('../train_results')
        res_dir.mkdir(parents=True, exist_ok=True)

        # train(path_to_train=imagePath, data_frame=df, pretrained_weights=weight, save_dir=res_dir, batch_size=BATCH_SIZE, shape=SHAPE, lr=1e-3, val_ratio=VAL_RATIO, epochs=20)
        # train(path_to_train=imagePath, data_frame=df, pretrained_weights=weight, save_dir=res_dir, batch_size=BATCH_SIZE, shape=SHAPE, lr=1e-4, val_ratio=VAL_RATIO, epochs=20)
        train(path_to_train=imagePath, data_frame=df, pretrained_weights=weight, save_dir=res_dir, batch_size=BATCH_SIZE, shape=SHAPE, lr=1e-5, val_ratio=VAL_RATIO, epochs=20)