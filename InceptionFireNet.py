
from keras.layers import *
from keras.models import *
from keras.utils import *
import numpy as np
from tensorflow.keras.optimizers import Adam

sq1x1 = "squeeze1x1"
exp1x1 = "expand1x1"
exp3x3 = "expand3x3"
relu = "relu_"

def fire_module(x, fire_id, squeeze=16, expand=64):

    s_id = 'fire' + str(fire_id) + '/'   
    x = Conv1D(squeeze, kernel_size =1, padding='same', name=s_id + sq1x1)(x)
    x = Activation('relu', name=s_id + relu + sq1x1)(x)

    left = Conv1D(expand, kernel_size =1, padding='same', name=s_id + exp1x1)(x)
    left = Activation('relu', name=s_id + relu + exp1x1)(left)

    right = Conv1D(expand,kernel_size =3, padding='same', name=s_id + exp3x3)(x)
    right = Activation('relu', name=s_id + relu + exp3x3)(right)

    x=concatenate([left, right]) #,name=s_id + 'concat'
    return x
    

def inception_module(x,
                     filters_1x1,
                     filters_3x3_reduce,
                     filters_3x3,
                     filters_5x5_reduce,
                     filters_5x5,
                     filters_pool_proj,
                     name=None):
    
    conv_1x1 = Conv1D(filters_1x1, kernel_size =1, padding='same', activation='relu')(x)
    
    conv_3x3 = Conv1D(filters_3x3_reduce, kernel_size =1, padding='same', activation='relu')(x)
    conv_3x3 = Conv1D(filters_3x3, kernel_size =3, padding='same', activation='relu')(conv_3x3)

    conv_5x5 = Conv1D(filters_5x5_reduce, kernel_size =1, padding='same', activation='relu')(x)
    conv_5x5 = Conv1D(filters_5x5, kernel_size =5, padding='same', activation='relu')(conv_5x5)

    pool_proj = MaxPool1D(3, strides=1, padding='same')(x)
    pool_proj = Conv1D(filters_pool_proj, kernel_size =1, padding='same', activation='relu')(pool_proj)

    output = concatenate([conv_1x1, conv_3x3,conv_5x5, pool_proj])#, conv_5x5 , 
    
    return output

def InceptionFireBlock(input_tensor,squeeze=16, expand=32,name=0):  
  x_pool = Conv1D(expand, kernel_size =3, strides=2, padding='same',activation='relu')(input_tensor)#, activation='relu', kernel_size = (1,1)
  x_pool = BatchNormalization()(x_pool)

  inc = inception_module(x_pool,
                     filters_1x1=squeeze,
                     filters_3x3_reduce=squeeze,
                     filters_3x3=expand,
                     filters_5x5_reduce=squeeze,
                     filters_5x5=expand,
                     filters_pool_proj=squeeze,
                     name="inception"+str(name))  
  
  inc_con= Concatenate()([x_pool, inc])

  se = fire_module(inc_con, fire_id=name, squeeze=squeeze, expand=expand)
  x = Concatenate()([x_pool,inc,se])
  return x

def InceptionFireModule(input_shape, nb_class, depth=3):
  xin = Input(shape= input_shape)
  
  x = Conv1D(16, kernel_size = 3, strides= 1, padding = 'same', activation='relu')(xin) 
  x = BatchNormalization()(x)

  x = Conv1D(16, kernel_size =3, strides= 1, padding = 'same', activation='relu')(x)
  x = BatchNormalization()(x)
  Nf=16
  for w in range(0,3):
    x = InceptionFireBlock(input_tensor=x, squeeze=(Nf*np.power(2,w))//2,expand=Nf*np.power(2,w),name=w)
  x = GlobalAveragePooling1D()(x)
  #x=Flatten()(x)
  dense_256 = Dense(256, activation='relu',name='dense_256')(x)
  dense_128 = Dense(128, activation='relu',name='dense_128')(x)
  dense_64 = Dense(64, activation='relu',name='dense_64')(dense_128)

  x = Dense(nb_class, activation= 'softmax')(dense_64)  
  model = Model(xin, x)

  model.compile(loss='binary_crossentropy', optimizer = Adam(lr = 1e-3), metrics = ['accuracy'])
  #loss='sparse_categorical_crossentropy' for multi-class datasets
  return model
