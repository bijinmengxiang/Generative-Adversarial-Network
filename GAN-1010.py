# -*- coding: utf-8 -*-
"""
Created on Fri May 21 08:40:44 2021

@author: cnhhdn
"""
import torch
import torch.nn as nn
import pandas
import matplotlib.pyplot as plt 
import random

learning_rate=0.1

#生成真实的数据集（喂给鉴别器）
def generate_realdata():
    real_data = torch.FloatTensor([
        random.uniform(0.9, 1.1),
        random.uniform(-0.1, 0.1),
        random.uniform(0.9, 1.1),
        random.uniform(-0.1, 0.1)
        ]);
    return real_data

def generate_randomdata(size):
    return torch.rand(size)

#构建鉴别器
class Discriminator(nn.Module):
    #构造函数，初始化父类
    def __init__(self):
        super().__init__()
        self.progress = []
        self.linear1 = nn.Linear(4, 3)
        self.sigmoid = nn.Sigmoid()
        self.linear2 = nn.Linear(3, 1)
        #损失函数
        self.loss_function = nn.MSELoss()
        #创建优化器，梯度下降法训练
        self.optimiser = torch.optim.SGD(self.parameters(),learning_rate)
           
    #前向传播方向    
    def forward(self,x):
        x = self.linear1(x)
        x = self.sigmoid(x)
        x = self.linear2(x)
        x = self.sigmoid(x)
        return x
    
    def train(self,data,target):
        if torch.cuda.is_available() :
            target = target.cuda()
            data = data.cuda()
        #通过网络，计算结果
        outputs = model(data)
        #计算损失值
        loss = self.loss_function(outputs,target)
        #定义记录方法（每十次记录一次loss值）
        if (counter % 10 == 0):
            self.progress.append(loss.item())
        if (counter % 1000 == 0):
            print("counter = ",counter)    
        #将梯度归零 进行反向传播 并更新权重    
        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()

    def plot_progress(self):
        df = pandas.DataFrame(self.progress,columns=['loss'])
        df.plot(ylim=(0,1.0),figsize=(16,8),alpha=0.1,marker='.',grid=True,yticks=(0,0.25,0.5))
        
#构造生成器
class Generator(nn.Module):
    def __init__(self):
        super().__init__()   
        self.progress = []
        self.linear1 = nn.Linear(1, 3)
        self.sigmoid = nn.Sigmoid()
        self.linear2 = nn.Linear(3, 4)
        #创建优化器，梯度下降法训练
        self.optimiser = torch.optim.SGD(self.parameters(),learning_rate)

    #前向传播方向    
    def forward(self,x):
        x = self.linear1(x)
        x = self.sigmoid(x)
        x = self.linear2(x)
        x = self.sigmoid(x)
        return x
    
    def train(self,model,data,target):
        if torch.cuda.is_available() :
            model = model.cuda()
            target = target.cuda()
            data = data.cuda()
        g_output = self.forward(data)
        d_output = model.forward(g_output)
        loss = model.loss_function(d_output,target)
        
        if (counter % 10 == 0):
            self.progress.append(loss.item())
        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()  
    
    def plot_progress(self):
        df = pandas.DataFrame(self.progress,columns=['loss'])
        df.plot(ylim=(0,1.0),figsize=(16,8),alpha=0.1,marker='.',grid=True,yticks=(0,0.25,0.5))
        

#训练
counter = 0

model = Discriminator()
Generator = Generator()

if torch.cuda.is_available() :
        model = model.cuda()
        Generator = Generator.cuda()
for i in range(10000):
    model.train(generate_realdata(),torch.ones(1))
    model.train(generate_randomdata(4),torch.zeros(1))
    model.train(Generator.forward(torch.FloatTensor([0.5]).cuda()).detach(),torch.zeros(1))
    Generator.train(model,torch.FloatTensor([0.5]),torch.FloatTensor([1.0]))
    counter = i
model.plot_progress()
Generator.plot_progress()

Generator = Generator.cpu()
model = model.cpu()
print(Generator.forward(torch.FloatTensor([0.5])))
print(model.forward(Generator.forward(torch.FloatTensor([0.5]))))
