import torch.nn.functional as F
import torch
import torch 
import torch.nn as nn
from torch.autograd import Variable
import torchvision.models as models
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import torch.optim as optim
import os
import cv2

# rootpath for dataset
root = '/data/wl/autofocus/learn2focus/dataset/'

def default_loader(path):
	return cv2.imread(path, cv2.IMREAD_GRAYSCALE)

# make my dataset
class MyDataset(Dataset):
	# initialize
	def __init__(self,txt, transform=None,target_transform=None, loader=default_loader):
		super(MyDataset,self).__init__()
		fh = open(txt, 'r')
		imgs = []
		for line in fh: 
			line = line.strip('\n')
			line = line.rstrip('\n')
			words = line.split()
			# patch path and label
			imgs.append((words[0],int(words[1])))      
		self.imgs = imgs
		self.transform = transform
		self.target_transform = target_transform
		self.loader = loader        
        
	def __getitem__(self, index):
		fn, label = self.imgs[index]
		img = self.loader(fn)
		img = np.dstack((img[:,:128],img[:,128:]))
		# single slice
		# five channel: two with dp-data; two with coordinates; one with lens postion
		x_idx = int(fn[-11:-9])
		y_idx = int(fn[-14:-12])
		x1 = x_idx * 40
		y1 = y_idx * 40
		x_dim = 1512
		y_dim = 2016
		x_list = x1 + np.arange(128)
		y_list = y1 + np.arange(128)
		# normalize to [-1,1]
		x_list_ = x_list.astype(np.float32)/(x_dim-1)*2-1
		x_list_channel = np.tile(x_list_.reshape(1,128), (128,1))
		y_list_ = y_list.astype(np.float32)/(y_dim-1)*2-1
		y_list_channel = np.tile(y_list_.reshape(128,1), (1,128))
		len_dim = 49
		len_idx = int(fn[-6:-4])
		len_idx_ = len_idx/(len_dim - 1)
		len_channel = len_idx_ * np.ones((128,128), dtype=np.float32)
		img = np.dstack((img, x_list_channel, y_list_channel, len_channel))
		
		if self.transform is not None:
			img = self.transform(img)
			
		return img, label #return
	
	def __len__(self):
		return len(self.imgs)