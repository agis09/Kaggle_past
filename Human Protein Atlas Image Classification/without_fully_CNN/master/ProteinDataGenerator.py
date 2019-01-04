
# credits: https://github.com/keras-team/keras/blob/master/keras/utils/data_utils.py#L302
# credits: https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
import numpy as np
import cv2
import keras.utils
from scipy import ndimage
broken_file_names = ['a9125fa6-bbbb-11e8-b2ba-ac1f6b6435d0','5297c0e2-bbc2-11e8-b2bb-ac1f6b6435d0','6c6ac5ea-bba0-11e8-b2b9-ac1f6b6435d0','edf568ca-bb9d-11e8-b2b9-ac1f6b6435d0','6282fe1e-bbc1-11e8-b2bb-ac1f6b6435d0','fcde8c06-bbb4-11e8-b2ba-ac1f6b6435d0','f1c7702c-bbc7-11e8-b2bc-ac1f6b6435d0','0afda11a-bba0-11e8-b2b9-ac1f6b6435d0','8ba4bc58-bbb5-11e8-b2ba-ac1f6b6435d0','6a82276e-bbc8-11e8-b2bc-ac1f6b6435d0','7ccb60c0-bbc8-11e8-b2bc-ac1f6b6435d0','bd7be178-bbb2-11e8-b2ba-ac1f6b6435d0']

class ProteinDataGenerator(keras.utils.Sequence):
    """
    注意 引数としてis_maskを追加。これがTrueなら出力をマスク画像とする。
    引数labelsは実際にonehotのラベルを入力とする。(これまではマスク画像を入力にしていた。)
    引数としてcrop_shapeを追加。2x2の配列を代入。
    """
    def __init__(self, paths, labels, batch_size, shape, is_mask, shuffle=False, use_cache=False, augment=True, crop_shape=None):
        self.paths, self.labels = paths, labels
        self.batch_size = batch_size
        self.shape = shape
        self.is_mask = is_mask
        self.shuffle = shuffle
        self.use_cache = use_cache
        self.augment = augment
        self.crop_shape = crop_shape
        if use_cache == True:
            self.cache = np.zeros((paths.shape[0], shape[0], shape[1], shape[2]), dtype=np.float16)
            self.is_cached = np.zeros((paths.shape[0]))
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.paths) / float(self.batch_size)))

    def __getitem__(self, idx):
        indexes = self.indexes[idx * self.batch_size: (idx + 1) * self.batch_size]

        paths = self.paths[indexes]
        Y = self.labels[indexes]
        X = np.zeros((paths.shape[0], self.shape[0], self.shape[1], self.shape[2]))
        y=[]
        # Generate data
        """
        Don't use "use_cache"
        """
        if self.use_cache == True:
            X = self.cache[indexes]
            for i, path in enumerate(paths[np.where(self.is_cached[indexes] == 0)]):
                if path.split('/')[-1] in broken_file_names:
                    continue
                image, y[i] = self.__load_image(path, y[i])
                self.is_cached[indexes[i]] = 1
                self.cache[indexes[i]] = image
                X[i] = image
        else:
            for i, path in enumerate(paths):
                if path.split('/')[-1] in broken_file_names:
                    continue
                X[i],tmp  = self.__load_image(path, Y[i])
                y.append(np.array(tmp))
            y=np.reshape(y,(len(y),y[0].shape[0],y[0].shape[1],y[0].shape[2]))
        if self.augment == True:
            num = X.shape[0]
            x_flip_indices = np.random.choice([0,1],num) #x軸方向の反転
            y_flip_indices = np.random.choice([0,1],num) #y軸方向の反転
            X = np.array([X[i,:,::-1,:] if x_flip_indices[i] == 1 else X[i,:,:,:] for i in range(num)])
            X = np.array([X[i,::-1,:,:] if y_flip_indices[i] == 1 else X[i,:,:,:] for i in range(num)])
            if self.is_mask:
                y = np.array([y[i,:,::-1,:] if x_flip_indices[i] == 1 else y[i,:,:,:] for i in range(num)])
                y = np.array([y[i,::-1,:,:] if y_flip_indices[i] == 1 else y[i,:,:,:] for i in range(num)])

            rotate_indices = np.random.choice([0,1,2,3],num)
            for i in range(3):
                X = np.array([np.rot90(X[j,:,:,:],k=1) if rotate_indices[j] > i else X[j,:,:,:] for j in range(num)])
                if self.is_mask:
                    y = np.array([np.rot90(y[j,:,:,:],k=1) if rotate_indices[j] > i else y[j,:,:,:] for j in range(num)])
        return [np.reshape(X[:, :, :, i], (X.shape[0], self.shape[0], self.shape[1], 1)) for i in range(4)], y

    def on_epoch_end(self):

        # Updates indexes after each epoch
        self.indexes = np.arange(len(self.paths))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __iter__(self):
        """Create a generator that iterate over the Sequence."""
        for item in (self[i] for i in range(len(self))):
            yield item

    def __load_image(self, path, label):
        # R = Image.open(path + '_red.png')
        # G = Image.open(path + '_green.png')
        # B = Image.open(path + '_blue.png')
        # Y = Image.open(path + '_yellow.png')

        R = cv2.imread(path + '_red.tif', cv2.IMREAD_GRAYSCALE)
        G = cv2.imread(path + '_green.tif', cv2.IMREAD_GRAYSCALE)
        B = cv2.imread(path + '_blue.tif', cv2.IMREAD_GRAYSCALE)
        Y = cv2.imread(path + '_yellow.tif', cv2.IMREAD_GRAYSCALE)
        im_shape = R.shape[0]

        im = np.stack((
            np.array(R),
            np.array(G),
            np.array(B),
            np.array(Y)), -1)

        # im = cv2.resize(im, (self.shape[0], self.shape[1]))
        if not self.crop_shape is None:
            top = np.random.randint(0,im_shape - self.crop_shape[0])
            left = np.random.randint(0, im_shape - self.crop_shape[1])
            bottom = top + self.crop_shape[0]
            right = left + self.crop_shape[1]
            if bottom > im.shape[0] or right > im.shape[1]:
                raise Exception('はみ出しています!')
            im = im[top:bottom, left:right, :]

        label = self.make_mask(G, label, size=32) if self.is_mask else label

        im = np.divide(im, 255.)
        return im, label

    def make_mask(self, img_g, img_class, size=32):
        """
        引数
        img_g : 緑画像
        img_label : 画像のクラスラベル(one hotのnp.array)

        出力
        img_label : ラベルshapeが(32,32)のnp.array
        """
        img_g = np.divide(img_g, 255.)
        # ガウシアンフィルター
        img_g=ndimage.gaussian_filter(img_g,5)
        # 画像の二値化
        ret,img_g = cv2.threshold(img_g,(np.amin(img_g)+np.amax(img_g))/10,1.,cv2.THRESH_BINARY)
        # img_g = cv2.adaptiveThreshold(img_g,1.,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
        # 画像の膨張
        img_g=cv2.dilate(img_g,(2,2),iterations=1)
        img_g=cv2.resize(img_g,(32,32))

        img_tmp=np.zeros((32,32))

        for i in range(28):
            if img_class[i]>0:
                mask = np.dstack((mask,img_g)) if i > 0 else img_g
            elif img_class[i]==0:
                mask = np.dstack((mask,img_tmp)) if i > 0 else img_tmp
        return mask

   

