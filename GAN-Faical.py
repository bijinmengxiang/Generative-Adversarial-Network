# -*- coding: utf-8 -*-
"""
Created on Sun May 30 15:04:28 2021

@author: cnhhdn
"""
import torch
import torchvision
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.hub import load_state_dict_from_url
import random
import numpy as np
import os, shutil
from PIL import Image
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import csv
from torchvision import transforms
import pandas

learning_rate=0.0001
train_transform = transforms.Compose([
        transforms.ToTensor()
    ])

batch_size_train=100
#使用ImageFolder库将图片转换为数据集
train_dataset =torchvision.datasets.ImageFolder(root='F:/train_sources/celeba',transform=train_transform)

#加载上方数据集，参数介绍在mnist_bumyself中注释有介绍
train_loader =DataLoader(train_dataset,batch_size=batch_size_train, shuffle=True,num_workers=0)

def generate_randomdata_seed_prove(batch,channels,height,width):
    random_data = torch.randn(batch,channels,height,width)
    return random_data

def generate_randomdata_seed(size):
    random_data = torch.randn(batch_size_train,size)
    return random_data
def generate_randomdata_seed_signal(size):
    random_data = torch.randn(size)
    return random_data
 

class Discriminator(nn.Module):
    count = 0
    #构造函数，初始化父类
    def __init__(self):
        super().__init__()
        self.progress = []
        self.linear1 = nn.Linear(3*218*178, 100)
        self.relu = nn.LeakyReLU()
        self.norm = nn.LayerNorm(100)
        self.linear2 = nn.Linear(100, 1)
        self.sigmoid = nn.Sigmoid()
        #损失函数
        self.loss_function = nn.BCELoss()
        #创建优化器，梯度下降法训练
        #self.optimiser = torch.optim.SGD(self.parameters(),learning_rate)
        self.optimiser = torch.optim.Adam(self.parameters(),lr = learning_rate)
           
    #前向传播方向    
    def forward(self,x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.norm(x)
        x = self.linear2(x)
        x = self.sigmoid(x)
        return x
    
    def train(self,data,target):
        self.count+=1
        if torch.cuda.is_available() :
            target = target.cuda()
            data = data.cuda()
        #通过网络，计算结果
        #print(data.shape)
        data = torch.flatten(data,1)
        #print(data.shape)
        #print(abc)
        outputs = model(data.float())
        #计算损失值
        self.loss_function = self.loss_function.cuda()
        loss = self.loss_function(outputs,target.float())
        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()
        #定义记录方法（每十次记录一次loss值）
        if (self.count % 2 == 0):
            self.progress.append(loss.item())
            #print("counter = ",self.count)
        if (self.count % 200 == 0):
            print("counter = ",self.count)
            print(loss)
    #将梯度归零 进行反向传播 并更新权重     
    def plot_progress(self):
        df = pandas.DataFrame(self.progress,columns=['loss'])
        df.plot(ylim=(0),figsize=(16,8),alpha=0.1,marker='.',grid=True,yticks=(0,0.25,0.5, 1.0, 5.0))
        
        
        
class Generator(nn.Module):
    countg = 0
    def __init__(self):
        super().__init__()   
        self.progress = []
        self.linear1 = nn.Linear(100, 300)
        self.relu = nn.LeakyReLU()
        self.norm = nn.LayerNorm(300)
        self.sigmoid = nn.Sigmoid()
        self.linear2 = nn.Linear(300, 3*218*178)
        #创建优化器，梯度下降法训练
        #self.optimiser = torch.optim.SGD(self.parameters(),0.1)
        self.optimiser = torch.optim.Adam(self.parameters(),learning_rate)


    #前向传播方向    
    def forward(self,x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.norm(x)
        x = self.linear2(x)
        x = self.sigmoid(x)
        return x
    
    def train(self,model,data,target):
        self.countg+=1
        if torch.cuda.is_available() :
            model = model.cuda()
            target = target.cuda()
            data = data.cuda()
        g_output = self.forward(data)
        d_output = model.forward(g_output)
        loss1 = model.loss_function(d_output,target)
        self.optimiser.zero_grad()
        loss1.backward()
        self.optimiser.step()  
        
        if (self.countg % 2 == 0):
            self.progress.append(loss1.item())
        if (self.countg % 400 == 0):
            print("countg = ",self.countg)
            print(loss1)
        
    
    def plot_progress(self):
        df = pandas.DataFrame(self.progress,columns=['loss'])
        df.plot(ylim=(0,1.0),figsize=(16,8),alpha=0.1,marker='.',grid=True,yticks=(0,0.25,0.5,1.0, 5.0))
        
     
model = Discriminator()
model = model.cuda()
Generator = Generator()
Generator = Generator.cuda()
"""
output = Generator.forward(generate_randomdata_seed_signal(100).cuda())
print(output.shape)
img = output.detach().cpu().view(218,178,3).numpy()
plt.imshow(img, interpolation='none', cmap='Blues')
"""
for i in range(10):
    for data,target in train_loader:
        #使用真数据对鉴别器进行训练
        model.train(data.cuda(),torch.ones(batch_size_train,1).cuda())
        #使用假数据对鉴别器进行训练
        model.train(generate_randomdata_seed_prove(batch_size_train,3,218,178).cuda(),torch.zeros(batch_size_train,1).cuda())
        #使用假数据计算鉴别器loss值后 对生成器进行训练
        Generator.train(model,generate_randomdata_seed(100),torch.ones(batch_size_train,1).cuda())

model.plot_progress()
Generator.plot_progress()

f, axarr = plt.subplots(2,3, figsize=(16,8))
for i in range(2):
    for j in range(3):
        output = Generator.forward(generate_randomdata_seed_signal(100).cuda())
        img = output.detach().cpu().view(218,178,3).numpy()
        axarr[i,j].imshow(img, interpolation='none', cmap='Blues')
        pass
    pass
