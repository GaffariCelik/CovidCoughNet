import pandas as pd
import numpy as np
import librosa, librosa.display, os, csv
import matplotlib.pyplot as plt


shift_orani="4"#10  #ALL_AUG_4
augmentedSignals = "Data/COUGHVID_audio_pitch_shift_"+str(shift_orani)+"/"
labels_path = "Data/COUGHVID_labels_audio_pitch_shift_"+str(shift_orani)+".csv" #abels_two_class
file_new_extended="Data/COUGHVID_data_new_audio_pitch_shift_"+str(shift_orani)+".csv"

train_csv = pd.read_csv(labels_path)
tot_rows = train_csv.shape[0]

header = 'filename chroma_stft rmse spectral_centroid spectral_bandwidth rolloff zero_crossing_rate'
for i in range(1, 21):
    header += f' mfcc{i}'
header += ' label'
header = header.split()

'''
Function which writes data to csv.
Features it uses: 
- RMSE 
- Chroma STFT
- Spectral Centroid
- Spectral Bandwidth 
- Spectral Rolloff 
- Zero Crossing 
Input: files in directory, 
'''
    

file = open(file_new_extended, 'w')
with file:
    writer = csv.writer(file)
    writer.writerow(header)
#data_new_extended = pd.read_csv('../input/coughclassifier-trial/data_new_extended.csv')
#print ('data_new_extended\n',data_new_extended)
for i in range(tot_rows):#10):#
        source = train_csv['uid'][i]
        file_name = augmentedSignals+source
        print ('source',source)        
        label =  train_csv['class'][i]
        print ('\nlabel', label)
        y,sr = librosa.load(file_name, mono=True, duration=5)
        chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
        rmse = librosa.feature.rms(y=y)
        spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr,hop_length=1024)
        spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr,hop_length=1024)
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr,hop_length=1024) #Nên có hop-length
        zcr = librosa.feature.zero_crossing_rate(y)
        mfcc = librosa.feature.mfcc(y=y, sr=sr)
        to_append = f'{np.mean(chroma_stft)} {np.mean(rmse)} {np.mean(spec_cent)} {np.mean(spec_bw)} {np.mean(rolloff)} {np.mean(zcr)}'    
        librosa.display.specshow(mfcc, x_axis='time') #Show MFCC
        #print(aa)
        for e in mfcc:
            to_append += f' {np.mean(e)}'
        to_append += f' {label}'
        value = [str(source)]
        value.extend(to_append.split())
        file = open(file_new_extended, 'a')
        with file:
            writer = csv.writer(file)
            writer.writerow(value)
            
