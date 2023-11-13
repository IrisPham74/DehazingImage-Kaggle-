from google.colab import drive
drive.mount('/content/drive')

import torch
import torch.nn as nn
import numpy as np
import torchvision

from torchvision.models import vgg16, VGG16_Weights

# %cd /content/drive/MyDrive/AIDTransformer

from dataset import *
from torch.utils.data import DataLoader

Dataset = '/content/drive/MyDrive/AIDTransformer/Dataset/Training_data/RICE'

data_loader = DataLoaderTrain(rgb_dir=Dataset, img_options={'patch_size': 256})

# !pip install timm

# !pip install einops

from model import *
model = Network()

# ----------------------- perpectual loss ----------------------
#model = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)

# Or use the most up-to-date weights
vgg_model = vgg16(weights=VGG16_Weights.DEFAULT)

if torch.cuda.is_available():
    vgg_model.cuda()

from collections import namedtuple

LossOutput = namedtuple("LossOutput", ["relu1_2", "relu2_2", "relu3_3"])

class LossNetwork(torch.nn.Module):
    def __init__(self, vgg_model):
        super(LossNetwork, self).__init__()
        self.vgg_layers = vgg_model.features
        self.layer_name_mapping = {
            '3': "relu1_2",
            '8': "relu2_2",
            '15': "relu3_3"
        }

    def forward(self, x):
        output = {}
        for name, module in self.vgg_layers._modules.items():
            x = module(x)
            if name in self.layer_name_mapping:
                output[self.layer_name_mapping[name]] = x
        return LossOutput(**output)

# Assuming output and ground_truth are your output and ground truth images
loss_network = LossNetwork(vgg_model)
loss_network.eval()
def perpectual_loss(y_pred, y_train):
  output_features = loss_network(y_pred)
  ground_truth_features = loss_network(y_train)

  # Calculate the perceptual loss
  perceptual_loss = 0
  for output_feature, ground_truth_feature in zip(output_features, ground_truth_features):
      perceptual_loss += torch.nn.functional.l1_loss(output_feature, ground_truth_feature)
  return perceptual_loss

# ----------------------- Edge Loss -----------------------------
import cv2 as cv

def sobel_edge_loss(img):
    #print('sobel')
    #print(type(img))
    #img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    sobel_kernel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0).expand(-1, 3, -1, -1)
    sobel_kernel_y = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0).expand(-1, 3, -1, -1)

    sobel_kernel_x = sobel_kernel_x.to(img.device)
    sobel_kernel_y = sobel_kernel_y.to(img.device)



    edges_x = F.conv2d(img, sobel_kernel_x, padding=0)
    edges_y = F.conv2d(img, sobel_kernel_y, padding=0)

    edges = torch.sqrt(edges_x**2 + edges_y**2)

    return edges

def edge_loss(output, target):
    
    output_edges = sobel_edge_loss(output)
    target_edges = sobel_edge_loss(target)

    loss = F.l1_loss(output_edges, target_edges)

    return loss

# ------------------------------- L1 Loss and Total Loss ----------------------
import torch.nn.functional as F

L1_Loss = nn.L1Loss()
def loss_function(y_pred, y_train):
  loss1 = L1_Loss(y_pred, y_train)
  loss2 = edge_loss(y_pred, y_train)
  loss3 = perpectual_loss(y_pred, y_train)

  total_loss = 1*loss1 + 5*loss2 + 10*loss3
  return total_loss

# ---------------------- TRaining -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# !pip install tqdm
from tqdm import tqdm
import math

def expand2square(timg,factor=32.0):
    timg = timg.unsqueeze(0)
    _, _, h, w = timg.size()

    X = int(math.ceil(max(h,w)/float(factor))*factor)

    img = torch.zeros(1,3,X,X).type_as(timg) # 3, h,w
    mask = torch.zeros(1,1,X,X).type_as(timg)


    img[:,:, ((X - h)//2):((X - h)//2 + h),((X - w)//2):((X - w)//2 + w)] = timg
    mask[:,:, ((X - h)//2):((X - h)//2 + h),((X - w)//2):((X - w)//2 + w)].fill_(1.0)

    return img, mask

from torch.optim.lr_scheduler import CosineAnnealingLR

optimizer = torch.optim.Adam(model.parameters(), lr = 0.0002)
scheduler = CosineAnnealingLR(optimizer, T_max = 100, verbose = True)

epochs = 120
#final_loss = []
for i in range(epochs):
  #for ii in range(len(data_loader)):
  for ii, data_train in enumerate(tqdm(data_loader), 0):
    rgb_gt = data_train[0].unsqueeze(0).to('cuda')

    rgb_noisy, mask = expand2square(data_train[1].cuda(), factor=128)
    rgb_noisy, mask = rgb_noisy.to('cuda'), mask.to('cuda')

    optimizer.zero_grad()

    # print(rgb_gt)
    # Forward pass
    outputs = model(rgb_noisy)
    #outputs = model(rgb_noisy, 1 - mask)

    # Compute loss
    loss = loss_function(outputs, rgb_gt)
    # final_loss.append(loss)

    # Backward pass and optimize
    loss.backward()

    optimizer.step()

    if ii == 409:
      break

  scheduler.step()
  print("Epoch number: {} and the loss : {}".format(i,loss.item()))

  torch.save({
            'epoch': i,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            #'loss': loss,
            }, '/content/drive/MyDrive/AIDTransformer/checkpoint.pth')


