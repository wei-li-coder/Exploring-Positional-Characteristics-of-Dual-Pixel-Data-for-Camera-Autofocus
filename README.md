# Exploring-Positional-Characteristics-of-Dual-Pixel-Data-for-Camera-Autofocus
Replicating the results of paper "Exploring Positional Characteristics of Dual-Pixel Data for Camera Autofocus"

Folder 'preprocess_data' contains codes for preprocessing the public dataset and making my own dataset for train.

Folder 'train' contains codes for training with MobileNet_V2 in this dataset.

The experimental results (all with MobileNet-v2) on test set are shown as follows. (Best results)
| | Algorithm | # of input channels | # of steps | =0 | <=1 | <=2 | <=4 |
| :-: | :------: | :------: | :------: | :------: | :------: | :------: | :------: |
| D1 | ICCV23(lens PE + ROI PE) | 5 | 1 | 0.181 | 0.460 | 0.646 | 0.851 |
| D1 | ICCV23(lens PE + ROI PE) | 5 | 2 | 0.198 | 0.510 | 0.702 | 0.897 |
| D1 | ICCV23(lens PE) | 3 | 1 | 0.141 | 0.364 | 0.541 | 0.777 |
| D1 | Learning to AF | 98 | 1 | 0.163 | 0.429 | 0.617 | 0.839 |


