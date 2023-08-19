from tensorflow.keras.callbacks import *
from metrics_visualization import *
from InceptionFireNet import InceptionFireModule
from sklearn.model_selection import train_test_split
from Create_Feature_Vektor import create_Vektor
from DeepConvNet import *
import numpy as np
import pickle
import pandas as pd



def data_load(data_path,shift_no=4):
  from sklearn import preprocessing
  # data_new_pitch_shift_ #2_breathing-deep_data_new_pitch_shift_ #3_vowel-a_data_new_pitch_shift_
  data = pd.read_csv(data_path+'3_vowel-a_data_new_pitch_shift_'+str(shift_no)+'.csv')  
  all_data=data
  all_data['label'].replace([2],[1],inplace=True)
  y = all_data.loc[:,'label']
  x = all_data.drop(['filename', 'label'], axis=1)
  X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

  X_train=np.array(X_train)
  X_test=np.array(X_test)
  X_train_hepsi=np.array(x)
  y_train_hepsi=y

  return X_train, X_test,X_train_hepsi, y_train, y_test,y_train_hepsi

#X_train, X_test,X_train_hepsi, y_train, y_test,y_train_hepsi =data_load("../../Data/Coswara/Extracted_data/",shift_no=4)


def model_train(model,epochs,BS,X_train,y_train,X_test,y_test,model_name,check_pnt, shift_no):
    fname=check_pnt+"best_model_Coswara_pitch_shift"+str(shift_no)+"_"+model_name+".h5"  
    callbacks = ModelCheckpoint(fname, monitor="val_accuracy", mode="max",
                              save_best_only=True, verbose=1)#,save_freq=50*(train_generator.samples//BS))
    callbacks=[callbacks]

    print("xxxxxxxxxxxxxxxxx[INFO] training head...xxxxxxxxxxxxxxxxxxxx")
    history =model.fit(X_train,y_train,validation_data=(X_test,y_test),batch_size=BS, epochs=epochs,verbose=1,callbacks=callbacks)#),callbacks=callbacks
 
def train(data_path,epochs=100,BS=32,nb_class=1, depth=5,model_name="InceptionSE_Vowel_a_MinMax",model_DeepNet_name="DeepConvNet",check_pnt="checkpoint/Coswara/",shift_no=4):
       
    X_train, X_test,X_train_hepsi, y_train, y_test,y_train_hepsi =data_load(data_path,shift_no)
    X_train=X_train.reshape((X_train.shape[0],X_train.shape[1],1))
    X_train_hepsi=X_train_hepsi.reshape((X_train_hepsi.shape[0],X_train_hepsi.shape[1],1))
    X_test=X_test.reshape((X_test.shape[0],X_test.shape[1],1)) 

    input_shape = X_train_hepsi.shape[1:]
    model = InceptionFireModule(input_shape=input_shape, nb_class=1, depth=5)
    model_train(model,epochs,BS,X_train_hepsi,y_train_hepsi,X_test,y_test,model_name,check_pnt, shift_no)

    #Create Feature Vektor
    
    v_X_train, v_X_test =create_Vektor(model, X_train, X_test)
    input_shape = v_X_train.shape[1:]
    model = DeepConvNet(input_shape, nb_class, plot_model=False)
    model_train(model,epochs,BS,v_X_train,y_train,v_X_test,y_test,model_DeepNet_name,check_pnt, shift_no)
    

    

if __name__ == '__main__':

    check_pnt="checkpoint/Coswara/"
    shift_no='4'#10''#'Raw'#8#'ALL_AUG_4'#10#'Raw'#10
    model_DeepNet_name="DeepConvNet"
    
    model_name="InceptionSE_Vowel_a_MinMax"
    data_path="../../Data/Coswara/Extracted_data/"
    train(data_path=data_path,epochs=100,BS=32,nb_class = 1,depth=5,model_name=model_name,model_DeepNet_name="DeepConvNet",check_pnt=check_pnt,shift_no=4)
