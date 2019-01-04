from keras.layers import GlobalAveragePooling2D, Dense, Reshape, multiply, Conv2D, BatchNormalization, ReLU, Add, MaxPooling2D, Input, Activation
from keras.models import Model, load_model

def se_block(ch, layer, ratio=8):
    """
    keras.layersのうち
    GlobalAveragePooling2D, Dense, Reshape, multiplyを使用する。
    """
    z = GlobalAveragePooling2D()(layer)
    x = Dense(ch // ratio, activation='relu')(z)
    x = Dense(ch, activation='sigmoid')(x)
    x = Reshape((1, 1, ch))(x)
    layer = multiply([layer, x])
    return layer

def res_block(ch, layer):
    """
    Conv2D, BatchNormalization, ReLU, Add
    """
    layer = Conv2D(ch, (3, 3))(layer)
    layer = BatchNormalization()(layer)
    layer = ReLU()(layer)
    skip = layer

    layer = BatchNormalization()(layer)
    layer = Conv2D(ch, (3, 3), padding='same')(layer)
    layer = BatchNormalization()(layer)
    layer = ReLU()(layer)
    layer = Conv2D(ch, (3, 3), padding='same')(layer)
    layer = BatchNormalization()(layer)
    layer = ReLU()(layer)

    layer = se_block(ch, layer)
    layer = Add()([layer, skip])
    return layer

def ch_CNN(layer):
    """
    Conv2D, ReLU, MaxPooling2D
    """
    layer = Conv2D(64, (5, 5), padding='same')(layer)
    layer = ReLU()(layer)
    layer = MaxPooling2D(pool_size=(2, 2))(layer)
    layer = Conv2D(128, (3, 3), padding='same')(layer)
    layer = ReLU()(layer)
    layer = MaxPooling2D(pool_size=(2, 2))(layer)
    layer = Conv2D(256, (3, 3), padding='same')(layer)
    layer = ReLU()(layer)
    layer = MaxPooling2D(pool_size=(2, 2))(layer)
    return layer


def create_model(input_shape):
    """
    Input, Add, Conv2D, Activation
    Model
    """
    ch1 = Input(input_shape)
    ch2 = Input(input_shape)
    ch3 = Input(input_shape)
    ch4 = Input(input_shape)

    x1 = ch_CNN(ch1)
    x2 = ch_CNN(ch2)
    x3 = ch_CNN(ch3)
    x4 = ch_CNN(ch4)

    x = Add()([x1, x2, x3, x4])
    x = Conv2D(256, (3, 3), padding='same')(x)
    x = ReLU()(x)
    x = MaxPooling2D(pool_size=(2,2))(x)
    x = Activation('sigmoid')(x)
    x = Conv2D(28, (3, 3), padding='same')(x)
    model = Model([ch1, ch2, ch3, ch4], x)

    return model

def create_discriminant_model(input_shape, weights=None):
    """
    引数
    input_shape : タプル
    weights : h5ファイルのパス
    """
    ch1 = Input(input_shape)
    ch2 = Input(input_shape)
    ch3 = Input(input_shape)
    ch4 = Input(input_shape)

    x = create_model(input_shape)([ch1, ch2, ch3, ch4]) if weights is None else load_model(weights, compile=False)([ch1, ch2, ch3, ch4])
    x = GlobalAveragePooling2D()(x)
    x = Dense(28, activation='sigmoid')(x)
    model = Model([ch1, ch2, ch3, ch4], x)
    return model
