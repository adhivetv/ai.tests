import os
import sys
import time
import datetime
import argparse
import math

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
import torch.optim as optim

import glob
import random
import os
import numpy as np

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import matplotlib.patches as patches

from skimage.transform import resize

from config import *
from yolov3 import *

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

	
def parse_data_config(path):
    options = dict()
    options['gpus'] = '0,1,2,3'
    options['num_workers'] = '10'
    with open(path, 'r') as fp:
        lines = fp.readlines()
    for line in lines:
        line = line.strip()
        if line == '' or line.startswith('#'):
            continue
        key, value = line.split('=')
        options[key.strip()] = value.strip()
    return options


class ListDataset(Dataset):
    def __init__(self, list_path, img_size=416):
        with open(list_path, 'r') as file:
            self.img_files = file.readlines()
        self.label_files = [path.replace('images', 'labels').replace('.png', '.txt').replace('.jpg', '.txt') for path in self.img_files]
        self.img_shape = (img_size, img_size)
        self.max_objects = 50
    def __getitem__(self, index):
        img_path = self.img_files[index % len(self.img_files)].rstrip()
        img = np.array(Image.open(img_path))
        while len(img.shape) != 3:
            index += 1
            img_path = self.img_files[index % len(self.img_files)].rstrip()
            img = np.array(Image.open(img_path))
        h, w, _ = img.shape
        dim_diff = np.abs(h - w)
        # Upper (left) and lower (right) padding
        pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
        # Determine padding
        pad = ((pad1, pad2), (0, 0), (0, 0)) if h <= w else ((0, 0), (pad1, pad2), (0, 0))
        # Add padding
        input_img = np.pad(img, pad, 'constant', constant_values=128) / 255.
        padded_h, padded_w, _ = input_img.shape
        # Resize and normalize
        input_img = resize(input_img, (*self.img_shape, 3), mode='reflect')
        # Channels-first
        input_img = np.transpose(input_img, (2, 0, 1))
        # As pytorch tensor
        input_img = torch.from_numpy(input_img).float()
        label_path = self.label_files[index % len(self.img_files)].rstrip()
        labels = None
        if os.path.exists(label_path):
            labels = np.loadtxt(label_path).reshape(-1, 5)
            # Extract coordinates for unpadded + unscaled image
            x1 = w * (labels[:, 1] - labels[:, 3]/2)
            y1 = h * (labels[:, 2] - labels[:, 4]/2)
            x2 = w * (labels[:, 1] + labels[:, 3]/2)
            y2 = h * (labels[:, 2] + labels[:, 4]/2)
            # Adjust for added padding
            x1 += pad[1][0]
            y1 += pad[0][0]
            x2 += pad[1][0]
            y2 += pad[0][0]
            # Calculate ratios from coordinates
            labels[:, 1] = ((x1 + x2) / 2) / padded_w
            labels[:, 2] = ((y1 + y2) / 2) / padded_h
            labels[:, 3] *= w / padded_w
            labels[:, 4] *= h / padded_h
        filled_labels = np.zeros((self.max_objects, 5))
        if labels is not None:
            filled_labels[range(len(labels))[:self.max_objects]] = labels[:self.max_objects]
        filled_labels = torch.from_numpy(filled_labels)
        return img_path, input_img, filled_labels
    def __len__(self):
        return len(self.img_files)

	
cuda = torch.cuda.is_available()

os.makedirs('output', exist_ok=True)
os.makedirs('checkpoints', exist_ok=True)

# Get data configuration
train_path      = '/mnt/ssd1024/yolov3/pytorch-yolo-v3/coco/trainvalno5k.txt'

# Get hyper parameters
hyperparams     = cfg[0]
learning_rate   = float(hyperparams['learning_rate'])
momentum        = float(hyperparams['momentum'])
decay           = float(hyperparams['decay'])
burn_in         = int(hyperparams['burn_in'])

# Initiate model
model = YOLO(cfg)
model.apply(weights_init_normal)

model = model.cuda()

model.train()

# Get dataloader
dataloader = torch.utils.data.DataLoader(
    ListDataset(train_path),
    batch_size=8, shuffle=False, num_workers=0)

Tensor = torch.cuda.FloatTensor

optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, dampening=0, weight_decay=decay)

for epoch in range(30):
    for batch_i, (_, imgs, targets) in enumerate(dataloader):
        imgs = Variable(imgs.type(Tensor))
        targets = Variable(targets.type(Tensor), requires_grad=False)
        optimizer.zero_grad()
        loss = model(imgs, targets)
        loss.backward()
        optimizer.step()
        print('[Epoch %d/%d, Batch %d/%d] [Losses: x %f, y %f, w %f, h %f, conf %f, cls %f, total %f, recall: %.5f]' %
                                    (epoch, 30, batch_i, len(dataloader),
                                    model.losses['x'], model.losses['y'], model.losses['w'],
                                    model.losses['h'], model.losses['conf'], model.losses['cls'],
                                    loss.item(), model.losses['recall']))
        model.seen += imgs.size(0)
		
    if epoch % 1 == 0:
        model.save_weights('%s/%d.weights' % ('checkpoints', epoch))