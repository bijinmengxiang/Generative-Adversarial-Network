# -*- coding: utf-8 -*-
"""
Created on Sun May 23 15:42:56 2021

@author: cnhhdn
"""
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import pandas,numpy,random 
import matplotlib.pyplot as plt 
from torch.utils.data import DataLoader
import torchvision

batch_size_train = 50
learning_rate=0.0001


def generate_randomdata(size):
    return torch.rand(size)

def generate_randomdata_seed(size):
    random_data = torch.randn(batch_size_train,size)
    return random_data

def generate_randomdata_seed_prove(size):
    random_data = torch.randn(size)
    return random_data

train_loader = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST('./data/', train=True, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               
                             ])),batch_size= batch_size_train,shuffle=True)

class Discriminator(nn.Module):
    count = 0
    #构造函数，初始化父类
    def __init__(self):
        super().__init__()
        self.progress = []
        self.linear1 = nn.Linear(784, 200)
        self.relu = nn.LeakyReLU(0.02)
        self.norm = nn.LayerNorm(200)
        self.linear2 = nn.Linear(200, 1)
        self.sigmoid = nn.Sigmoid()
        #损失函数
        self.loss_function = nn.BCELoss()
        #创建优化器，梯度下降法训练
        #self.optimiser = torch.optim.SGD(self.parameters(),learning_rate)
        self.optimiser = torch.optim.Adam(self.parameters(),learning_rate)
           
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
        loss = self.loss_function(outputs,target.float())
        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()
        #定义记录方法（每十次记录一次loss值）
        if (self.count % 2 == 0):
            self.progress.append(loss.item())
        if (self.count % 400 == 0):
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
        self.linear1 = nn.Linear(100, 200)
        self.relu = nn.LeakyReLU(0.02)
        self.norm = nn.LayerNorm(200)
        self.sigmoid = nn.Sigmoid()
        self.linear2 = nn.Linear(200, 784)
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
Generator = Generator()
model = model.cuda()
Generator = Generator.cuda()
sum_count = 0

for i in range(5):
    for data,target in train_loader:
        #第一步：训练鉴别器，给鉴别器喂1或0数据
    
        model.train(data.cuda(),torch.ones(batch_size_train,1).cuda())
        #model.train(generate_randomdata(784),torch.zeros(1))
        #第二步：训练鉴别器，给鉴别器喂生成器产生的数据    
        model.train(Generator.forward(generate_randomdata_seed(100).cuda()).detach(),torch.zeros(batch_size_train,1))
        #第三步：训练生成器，将假数据喂给鉴别器后，保存loss，用此loss来训练生成器
        Generator.train(model,generate_randomdata_seed(100).cuda(),torch.ones(batch_size_train,1).cuda())
        
    
model.plot_progress()
Generator.plot_progress()
f, axarr = plt.subplots(2,3, figsize=(16,8))
for i in range(2):
    for j in range(3):
        output = Generator.forward(generate_randomdata_seed_prove(100).cuda())
        img = output.detach().cpu().numpy().reshape(28,28)
        axarr[i,j].imshow(img, interpolation='none', cmap='Blues')
        pass
    pass

        
        
        
        
        
