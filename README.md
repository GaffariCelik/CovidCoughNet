# CovidCoughNet
CovidCoughNet: A new method based on convolutional neural networks and deep feature extraction using pitch-shifting data augmentation for covid-19 detection from cough, breath, and voice signals

This repository contains the used source code presented in the paper CovidCoughNet: A new method based on convolutional neural networks and deep feature extraction using pitch-shifting data augmentation for covid-19 detection from cough, breath, and voice signals.

We used COUGHVID and Coswara datasets. The following operations must be performed in order for the code to work properly:


- Extraction_data.py--> Used to extract compressed signals.

- pitch_Shift_Coswara.py and pitch_Shift_COUGHVID.py --> By providing resampling of the audio used to get better sound quality.

- create_MFCC.py --> Feature extraction techniques such as square energy (RMSE), spectral centroid (SC), spectral bandwidth (SB), spectral rolloff (SR), zero crossing rate (ZCR), and Mel frequency cepstral coefficients (MFCC) were used. The feature extraction techniques provide an efficient computational method for the application for which it is designed, as well as extracting important features from cough, breath, and voice signals.

- InceptionFireNet.py--> Proposed CNN-based deep feature extraction module

- DeepConvNet.py--> Covid-19 detection module

- train_Model--> Training the proposed methods

- test_Model.py--> Test phase

- metrics_visualization.py--> plotROCCurve, plotConfusionMatrix, and metrics

  
# Citation and More Information:

Please check out our article for more detailed information.

Celik. G (2023). CovidCoughNet: A new method based on convolutional neural networks and deep feature extraction using pitch-shifting data augmentation for covid-19 detection from cough, breath, and voice signals. Computers in Biology and Medicine. 163, 107153.

doi: https://doi.org/10.1016/j.compbiomed.2023.107153.


