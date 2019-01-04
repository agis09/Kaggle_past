import numpy as np
import cv2
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from scipy import ndimage

def getDataSet(img_dir, input_data_frame, is_mask):
	"""
	引数
	path : 画像の含まれるディレクトリの絶対パス
	input_data_frame : IdとTargetをもつpandas.DataFrame
	is_mask : bool型で、Trueならラベルとしてマスク画像を返す。Falseなら複数クラス版one hotを返す。

	返り値
	paths :「 (親絶対パス) / 画像ファイルのヘッド名」という不完全なパスの配列
	labels : 対応するラベルの配列
	"""
	data = input_data_frame
	class_num = 28
	paths = []
	labels = []
	# nameはID(画像のヘッドネーム)で、lblは画像に含まれるクラスのリスト
	for name, lbl in tqdm(zip(data['Id'], data['Target'].str.split(' '))):
		# one_hotの列の配列を作り、それを足し算してクラスの複数クラス版onehotリストを作る。
		lbl = [int(l) for l in lbl]
		lbl_one_hot = np.identity(class_num)[lbl].sum(axis=0)
		im_name=str(img_dir)+'/'+name
		paths.append(im_name)
		if is_mask:
			labels.append(getMaskImage(im_name+'_green.png', lbl_one_hot))
		else:
			labels.append(lbl_one_hot)
	return np.array(paths), np.array(labels)


def getMaskImage(img_path, img_class):
	"""
	引数
	img_path : 画像のパス(pathlibオブジェクトか、string)
	img_label : 画像のクラスラベル(one hotのnp.array)

	出力
	img_label : ラベルshapeが(32,32)のnp.array
	"""
	img_g = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
	if img_g.shape is None:
		raise Exception('ERROR!!:画像'+str(img_g)+'の読み込みに失敗しました。')

	# 画像を正規化
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
	        img_label = np.dstack((img_label,img_g)) if i > 0 else img_g
	    elif img_class[i]==0:
	        img_label = np.dstack((img_label,img_tmp)) if i > 0 else img_tmp
	return img_label

def getValidationDataset(img_dir, input_data_frame, is_mask):
	return getDataSet(img_dir, input_data_frame, is_mask)


def getTrainDataset(img_dir, input_data_frame, is_mask):
	return getDataSet(img_dir, input_data_frame, is_mask)

def getTrainDataset100(img_dir, input_data_frame, is_mask):
	"""
	引数
	path : 画像の含まれるディレクトリの絶対パス
	input_data_frame : IdとTargetをもつpandas.DataFrame
	is_mask : マスクか否か

	返り値
	pathsTrain :「 (親絶対パス) / 画像ファイルのヘッド名」という不完全なパスの配列
	labelsTrain : pathsTrainに対応するラベル画像の配列
	"""
	data = input_data_frame
	class_num = 28
	paths = {i:[] for i in range(class_num)}
	labels = {i:[] for i in range(class_num)}
	#label_counter = np.zeros(class_num) # これまでに各クラスがそれぞれ何枚読み込まれたかを測るカウンター。
	# nameはID(画像のヘッドネーム)で、lblは画像に含まれるクラスのリスト
	for name, lbl in tqdm(zip(data['Id'], data['Target'].str.split(' '))):
		if len(lbl) > 1:
			continue
		class_index = int(lbl[0])
		# クラスラベルをone hotに変換
		lbl_one_hot = np.identity(class_num)[int(lbl[0])]
		# # 各クラスの累計数をカウントし、新たな追加対象を追加した際に100を超える場合はスキップ
		# if np.any(label_counter + lbl_one_hot > 100):
		# 	continue
		# else :
		# 	label_counter += lbl_one_hot

		if len(paths[class_index]) == 100:
			continue
		im_name=str(img_dir)+'/'+name
		paths[class_index].append(im_name)

		if is_mask:
			labels[class_index].append(getMaskImage(im_name+'_green.png', lbl_one_hot))
		else :
			labels[class_index].append(lbl_one_hot)


	paths = {i: repeat_1darray(paths[i], 100) for i in range(class_num)}
	paths = dict_to_reduced_list(paths)

	labels = {i: repeat_4darray(labels[i], 100) for i in range(class_num)} if is_mask else {i: repeat_2darray(labels[i], 100) for i in range(class_num)}
	labels = dict_to_reduced_list(labels)

	return np.array(paths), np.array(labels)

def dict_to_reduced_list(dictionary):
	reduced_list = []
	for i in dictionary.values():
		reduced_list.extend(i)
	return reduced_list

def repeat_1darray(input_array,length_out):
	"""
	引数
	input_array : 1次元配列
	length_out : 出力配列の長さ

	入力配列を好きな長さに引き伸ばす、または縮める関数。
	例：[1,2]を長さ4に引き延ばす場合は、[1,2,1,2]になる。
	例：[1,2]を長さ1に縮める場合は、[1]になる。
	"""
	input_length = np.array(input_array).shape[0]
	if input_length == length_out:
		return np.array(input_array)
	elif input_length == 0:
		print('リストが空です。そのまま返します。')
		return np.array(input_array)
	iters = -(-length_out // input_length) # 出力の配列長さを入力配列の長さで割って、繰り返し回数を算出
	tmp = np.tile(input_array,iters)
	out = tmp[:length_out]
	return out

def repeat_2darray(input_array,length_out):
	"""
	引数
	input_array : 3次元配列
	length_out : 出力配列の長さ
	"""
	input_length = np.array(input_array).shape[0]
	if input_length == length_out:
		return np.array(input_array)
	elif input_length == 0:
		print('リストが空です。そのまま返します。')
		return np.array(input_array)
	iters = -(-length_out // input_length) # 出力の配列長さを入力配列の長さで割って、繰り返し回数を算出
	tmp = np.tile(input_array,[iters,1])
	out = tmp[:length_out]
	return out

def repeat_4darray(input_array,length_out):
	"""
	引数
	input_array : 3次元配列
	length_out : 出力配列の長さ
	"""
	input_length = np.array(input_array).shape[0]
	if input_length == length_out:
		return np.array(input_array)
	elif input_length == 0:
		print('リストが空です。そのまま返します。')
		return np.array(input_array)
	iters = -(-length_out // input_length) # 出力の配列長さを入力配列の長さで割って、繰り返し回数を算出
	tmp = np.tile(input_array,[iters,1,1,1])
	out = tmp[:length_out]
	return out

def getTestDataset():
	"""
	未編集。
	"""
	path_to_test = DIR + 'test/'
	data = pd.read_csv(DIR + 'sample_submission.csv')

	paths = []
	labels = []

	for name in data['Id']:
	    y = np.ones(28)
	    paths.append(os.path.join(path_to_test, name))
	    labels.append(y)

	return np.array(paths), np.array(labels)
