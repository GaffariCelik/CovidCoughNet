import pandas as pd
import os
import librosa
import librosa.display
import cv2
import numpy as np
import soundfile as sf
import json

### Function for generatng pitch_shifted audio samples
data_dir = 'Extracted_data/cough_outputs' #cough_outputs
samples = os.listdir(data_dir)

samples = [sample for sample in samples if len(sample) == 28]
len(samples)

def crate_metadata(samples):
    Y = pd.DataFrame(columns = ['uid'])
    for i in range (len(samples)):
        Y = Y.append({'uid':samples[i]},ignore_index=True) 
    metaDataPath = "meta_data.csv"
    Y.to_csv(metaDataPath,index=False)
#crate_metadata(samples)
    

ind_paths = []
for i in samples:
    ind_paths.append(os.path.join(data_dir, i))
 
ind_paths
    
ind_cough_paths = []
for i in ind_paths:
    ind_cough_paths.append(os.path.join(i, 'vowel-a.wav')) # cough-heavy.wav breathing-deep

ind_labels = []


shift_orani="4"#10 #ALL_AUG_4
n_steps_orani=-4#
labels_path = "Extracted_data/3_vowel-a_labels_audio_shift_"+str(shift_orani)+".csv" #labels_audio_raw
augmentedSignals = "Extracted_data/3_vowel_a_audio_pitch_shift_"+str(shift_orani)+"/"  #audio_pitch_shift_4
Y = pd.DataFrame(columns = ['uid','class'])
counter = 0

#ind_paths.sort(reverse=True)
hatali_orneksys=0
for i in ind_paths:
    try:        
        signal , sr = librosa.load(i+"/vowel-a.wav") # cough-heavy # breathing-deep
        #print(signal.size)
        with open(os.path.join(i,'metadata.json'), 'r') as file:
            metadata = json.load(file)
            covid_status = metadata['covid_status']
            if covid_status.startswith('positive'):
                
                pitch_shifting = librosa.effects.pitch_shift(signal,sr,n_steps=-4)
                sf.write(augmentedSignals+"sample{0}_{1}.wav".format(counter,1),pitch_shifting, sr,'PCM_24')
                label ="sample{0}_{1}.wav".format(counter,1)
                Y = Y.append({'uid':label,'class':1},ignore_index=True)                
                counter+=1
    
                
            else:            
                pitch_shifting = librosa.effects.pitch_shift(signal,sr,n_steps=-4)
                sf.write(augmentedSignals+"sample{0}_{1}.wav".format(counter,0),pitch_shifting, sr,'PCM_24')
                label ="sample{0}_{1}.wav".format(counter,0)
                Y = Y.append({'uid':label,'class':0},ignore_index=True)                
                counter+=1
    except ValueError:
        hatali_orneksys+=1
        print ('Okuma HATASI!!!!!!!!!!!!!!!!!!!')
        continue                    
            
