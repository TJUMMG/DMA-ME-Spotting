# Duration-Aware and Mode-Aware Micro-Expression Spotting for Long Video Sequences
Copyrighht(c) 2024 Jing Liu

```
If you use this code, please cite the following publication:
J. Liu, X. Li, G. Zhai, "Duration-Aware-and-Mode-Aware-Micro-Expression-Spotting-for-Long-Video-Sequences", to appear in Displays.
```

## Contents

1. [Environment](#1)
2. [Test](#2)

<h3 id="1">Environment</h3>
The codes need to run in the environment: Python 3.7.

### Testing
Firstly, download the two datasets used for experiments: CAS(ME)^2 and SAMM Long Videos and put them in 'dataset' folder. Download “shape_predictor_68_face_landmarks.dat” from the following link: https://pan.baidu.com/s/1_Ml4D6lN1_1HeGAb7kGy1Q with password ‘msfd’ and place it under directory ‘D&M-A ME spotting\code\landmark_model’

Secondly, run the following codes to reproduce the LBP feature based results provided in the paper:

(1) Run the codes in the folder "LBP_feature_based/extract_features" to extract LBP features:
```
$ python extract_feature.py
```
(2) Calculate contrast differences. In the folder "LBP_feature_based/calculate_contrast_differences", run the following codes to calculate the contrast differences of the extracted LBP features by symmetric or asymmetric windows of CAS(ME)^2 and SAMM Long Videos:
```
$ python calculate_contrast_difference_symmetric_window.py  
$ python calculate_contrast_difference_asymmetric_window.py 
```
(3) Reproduce spotting results. 
In the folder "LBP_feature_based/test/CAS(ME)^2", run the following codes for testing CAS(ME)^2 dataset:

To reproduce the spotting results of single scale symmetric windows and single scale asymmetric windows, run the Python codes:
```
$ python single_scale_symmetric_window.py  
$ python single_scale_asymmetric_window.py
```
To reproduce the fusion results of multi-scale windows and multi-mode windows, run the Python codes:
```
$ python multiple_scales_fusion.py  
$ python multiple_modes_fusion.py  
```
To reproduce the fusion results of our proposed duration & mode-aware ME spotting method (multi-scale and multi-mode sliding windows), run the Python codes:
```
$ python D&M-A_ME_Spotting.py  
```

In the folder “LBP_feature_based/test/SAMM Long Videos”, runn the following codes for testing SAMM Long Videos dataset:

To reproduce the fusion results of our proposed duration & mode-aware ME spotting method (multi-scale and multi-mode sliding windows), run the Python codes:
```
$ python D&M-A_ME_Spotting.py 
```

Thirdly, run the following codes to reproduce the MDMO feature (SP-pattern) based results provided in the paper:

(1) Extract MDMO features. 
In the folder "MDMO_feature_based/extract_features", run the Python codes:
```
$ python get_OFs.py  % get optical flows for each frame of videos in CAS(ME)^2
$ python get_local_OFs.py  % get local optical flows by removing global movements
$ python extract_MDMO_feature.py  % extract the MDMO magnitude and angle features
```
(2) Reproduce spotting results
In the folder "MDMO_feature_based/test", run the following codes to reproduce the fusion results of our duration & mode-aware ME spotting method based on the model proposed by the MEGC2020 competition paper "Spatio-temporal fusion for Macro- and Micro-expression Spotting in Long Video Sequences":
```
$ python D&P-A_ME_Spotting.py 
```
