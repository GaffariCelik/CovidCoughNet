import pandas as pd
import os
import librosa
import librosa.display
import cv2
import numpy as np
import soundfile as sf

### Function for generatng pitch_shifted audio samples

shift_orani=4#10 #ALL_AUG_4
n_steps_orani=-4#*shift_orani
cough_detected=0.8
metaDataPath = "Data/meta_data.csv"
audioDataPath = "Data/wavs-silence-removed/"
augmentedSignals = "Data/COUGHVID_audio_pitch_shift_"+str(shift_orani)+"/"
labels_path = "Data/COUGHVID_labels_audio_pitch_shift_"+str(shift_orani)+".csv" #abels_two_class
    
def pitchShift():

    metaData = pd.read_csv(metaDataPath)
    counter = 0
    Y = pd.DataFrame(columns = ['uid','class'])
    for index,row in metaData.iterrows():
        fname = row["uuid"]
        print(fname, " ", str(index+1),"/",str(metaData.shape[0]))
        signal , sr = librosa.load(audioDataPath+fname+".wav")
        ## Cough detection refinment: greater than 0.7
        
        #if row["cough_detected"] >= cough_detected:
            ## Multi-class to binary classification:
        if row["status"]=="COVID-19": #row["status"]=="COVID-19":
            
            pitch_shifting = librosa.effects.pitch_shift(signal,sr,n_steps=-4)
            sf.write(augmentedSignals+"sample{0}_{1}.wav".format(counter,1),pitch_shifting, sr,'PCM_24')
            label ="sample{0}_{1}.wav".format(counter,1)
            Y = Y.append({'uid':label,'class':1},ignore_index=True)                
            counter+=1              
            
        elif row["status"] == "symptomatic":
            
            ####  pitch_shifting
            pitch_shifting = librosa.effects.pitch_shift(signal,sr,n_steps=-4)
            sf.write(augmentedSignals+"sample{0}_{1}.wav".format(counter,2),pitch_shifting, sr,'PCM_24')
            label ="sample{0}_{1}.wav".format(counter,2)
            Y = Y.append({'uid':label,'class':2},ignore_index=True)                
            counter+=1  
        else:
            ####  pitch_shifting
            pitch_shifting = librosa.effects.pitch_shift(signal,sr,n_steps=-4)
            sf.write(augmentedSignals+"sample{0}_{1}.wav".format(counter,0),pitch_shifting, sr,'PCM_24')
            label ="sample{0}_{1}.wav".format(counter,0)
            Y = Y.append({'uid':label,'class':0},ignore_index=True)                
            counter+=1   
             ####  pitch_shifting
            
    Y.to_csv(labels_path,index=False)


    
pitchShift()