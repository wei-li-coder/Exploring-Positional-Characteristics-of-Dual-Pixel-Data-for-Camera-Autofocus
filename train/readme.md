__model.py__ contains codes for MobileNetV2, which is based on official implementation of Pytorch, with width_mult = 1.0 and the input channel number = 5 (2 channels for dp data + 1 channel for lens position + 2 channel for patch coordinate). (https://github.com/pytorch/vision/blob/main/torchvision/models/mobilenetv2.py)

__train.py__ contains codes for training the model and make predictions.

__load_data.py__ contains codes for loading my dataset.
