from tensorflow.keras.callbacks import *
from metrics_visualization import *
from InceptionFireNet import InceptionFireModule
from sklearn.model_selection import train_test_split
from Create_Feature_Vektor import create_Vektor
from DeepConvNet import *
import numpy as np
import pickle
import pandas as pd

from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn import preprocessing




def data_load(data_path,shift_no=4):

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
  

def model_test(model,X_test,y_test,shift_no,model_name,grafik,cmp,check_pnt):
    fname=check_pnt+"best_model_Coswara_pitch_shift"+str(shift_no)+"_"+model_name+".h5"
    model.load_weights(fname)
    loss, acc = model.evaluate(X_test, y_test, verbose=2)
    print("Restored model, accuracy: {:5.2f}%".format(100 * acc))
    y_pred = model.predict(X_test)
    # for binary classification
    y_pred[y_pred <= 0.5] = 0.
    y_pred[y_pred > 0.5] = 1.
    #
    plot_actual_vs_predicted(y_test,y_pred,"Test Data Predictions",grafik,cmp)
    Metric_Sensivity(y_test,y_pred) 
    metrics_auc=Metric_auc(y_test,y_pred) 
    Roccc_Curve(y_test,y_pred,metrics_auc)
  



def model_Testing(data_path, model_name,shift_no,grafik='normal',cmp='Cividis',nb_class=1,depth=5,check_pnt="checkpoint/Coswara/"):  
    
    X_train, X_test,X_train_hepsi, y_train, y_test,y_train_hepsi =data_load(data_path,shift_no)
    X_train=X_train.reshape((X_train.shape[0],X_train.shape[1],1))
    X_train_hepsi=X_train_hepsi.reshape((X_train_hepsi.shape[0],X_train_hepsi.shape[1],1))
    X_test=X_test.reshape((X_test.shape[0],X_test.shape[1],1)) 
    
    input_shape = X_train_hepsi.shape[1:]
    model = InceptionFireModule(input_shape=input_shape, nb_class=nb_class, depth=depth)
    
    v_X_train, v_X_test =create_Vektor(model, X_train, X_test)
    input_shape = v_X_train.shape[1:]
    model = DeepConvNet(input_shape, nb_class, plot_model=False)
    
    model_test(model,v_X_test,y_test,shift_no,model_name,grafik,cmp,check_pnt)
    
if __name__ == '__main__':
    
    check_pnt="checkpoint/Coswara/"
    data_path="../../Data/Coswara/Extracted_data/"
    model_name="DeepConvNet"
    model_Testing(data_path, model_name,shift_no='4',grafik='normal',cmp='Cividis',nb_class=1,depth=5,check_pnt=check_pnt)
