
from keras.layers import *
from keras.models import *
from keras.utils import *
import numpy as np
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

def get_Norm(x_input):

    x = BatchNormalization()(x_input)
    x = LeakyReLU()(x)
    return x

def get_Connectblock(layer, input_channels,output_channels):
    x_conv = Conv1D(input_channels, kernel_size =5, strides=2, padding='same')(layer)#kernel_size = (1,1)
    y=x_conv
    for i in range(4):
        x_conv = Conv1D(output_channels, kernel_size = 5, strides=1, padding='same')(x_conv)#kernel_size = (1,1)
        x_conv=get_Norm(x_conv)
    x = Concatenate()([x_conv, y])

    return x


def DeepConvNet(input_shape, num_classes, plot_model=False):
    xin = Input(shape= input_shape)
    x = Conv1D(32, kernel_size = 5, strides= 1, padding = 'same')(xin) #, activation='relu'
    x=get_Norm(x)

    x = get_Connectblock(x, 32,64)
    x = get_Connectblock(x, 64,128)
    x = get_Connectblock(x, 128,256)
    x = get_Connectblock(x, 256,512)

    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dense(64, activation='relu')(x)

    x = tf.keras.layers.Dense(num_classes, activation='sigmoid')(x) #softmax
    model = tf.keras.models.Model(inputs=xin, outputs=x)

    model.compile(loss='binary_crossentropy', optimizer = Adam(lr = 1e-3), metrics = ['accuracy'])#'binary_crossentropy','categorical_crossentropy'


    if plot_model:
        tf.keras.utils.plot_model(model, to_file='model.png', show_shapes=True)

    return model


