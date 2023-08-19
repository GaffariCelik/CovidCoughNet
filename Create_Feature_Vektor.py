
from tensorflow.keras.callbacks import ModelCheckpoint
from keras.models import *
import numpy as np


def create_Vektor(model,X_train,X_test):

  vector_layers = ['dense_128', 'dense_64']
  model_for_vector = Model(
      inputs=model.input,
      outputs=model.get_layer('dense_64').output
      #outputs=[model.get_layer(layer).output for layer in vector_layers]
  )

  v_X_train = model_for_vector.predict(X_train)
  v_X_train = np.array(v_X_train)

  v_X_test = model_for_vector.predict(X_test)
  v_X_test = np.array(v_X_test)
  
  v_X_train=v_X_train.reshape((v_X_train.shape[0],v_X_train.shape[1],1))
  v_X_test=v_X_test.reshape((v_X_test.shape[0],v_X_test.shape[1],1))  
  
  return v_X_train, v_X_test


 
