from keras.models import load_model
from PIL import Image
import pathlib
import numpy as np
import cv2
import argparse

def predict_imgs(paths, model):
        """
        入力：画像のパスのリスト、モデルオブジェクト
        出力：予測画像(4次元np.array)s
        """
        for num, path in enumerate(paths):
            with Image.open(path) as f:
                f.thumbnail((512,512), Image.ANTIALIAS)
                image = np.asarray(f.convert('L'), dtype=np.float32)
            if num ==0:
                images = image[np.newaxis,:,:,np.newaxis]
            else:
                image = image[np.newaxis,:,:,np.newaxis]
                images = np.r_[images,image]
        val = model.predict([images for i in range(4)])
        return val


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict all images in a directory.')
    parser.add_argument('input_dir', help='Write an absolute path to an input directory.', type=str)
    parser.add_argument('output_dir', help='Write an absolute path to an input directory.', type=str)
    args = parser.parse_args()

    if not pathlib.Path(args.output_dir).exists():
        pathlib.Path(args.output_dir).mkdir()

    model_name = '/home/elsa/kaggle/protein/kaggle_protein/pretrain_results/lr1e-05/next_base.model0.000010'
    model = load_model(model_name, compile=False)
    for count,image in enumerate(pathlib.Path(args.input_dir).glob('*_green*')):
        if count > 30:
            break
        headname = image.name.split('_green')[0]
        # 予測画像を格納するためのディレクトリを用意する
        if not pathlib.Path(args.output_dir,headname).exists():
            pathlib.Path(args.output_dir,headname).mkdir()
        suffix = image.suffix
        output_paths = [pathlib.Path(args.output_dir,headname,format(num)+suffix) for num in range(28)]
        predicted = predict_imgs([image], model)
        predicted *= 255
        #print(predicted.shape)
        for i in range(28):
            cv2.imwrite(str(output_paths[i]),predicted[0,:,:,i])
        #print('画像'+headname+'の予測が完了しました。')
    print('終了します。')