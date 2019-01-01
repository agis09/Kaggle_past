original_csv_path = '/home/elsa/Downloads/train.csv'
csv_path = '/home/elsa/Downloads/train_cut.csv'
train_path = '/home/elsa/Downloads/train_cut'
import pandas as pd
import pathlib
input = pd.read_csv(original_csv_path)
pd.DataFrame([['Id','Target']]).to_csv(csv_path, header=False, index=False)
# データフレームのiterrowsで行名（番号）と値（series）からなるタプルが与えられる
for line in input.iterrows():
	name, label = line[1].get_values()
	for file in pathlib.Path(train_path).glob(name+'*.tif'):
		basename = file.name
		pd.DataFrame([[basename, label]]).to_csv(csv_path, mode='a', header=False, index=False)
