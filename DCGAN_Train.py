# -*- coding: utf-8 -*-
from __future__ import print_function
from Utilities.SaveAnimation import Video
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import math
from IPython.display import HTML
import time
import pandas as pd
import pickle

#Get GPU Information#获取GPU信息
print("CUDA is available: {}".format(torch.cuda.is_available()))
print("CUDA Device Count: {}".format(torch.cuda.device_count()))
print("CUDA Device Name: {}".format(torch.cuda.get_device_name(0)))

#Location of Training Data#训练数据的位置
spectra_path = 'C:/.../absorptionData_HybridGAN.csv'

#Location to Save Models (Generators and Discriminators)#Location 保存模型（生成器和判别器）
save_dir = 'C:/.../'

#Root directory for dataset (images must be in a subdirectory within this folder)#数据集的根目录（图像必须在此文件夹的子目录中）
img_path = 'C:/.../Images'

def Excel_Tensor(spectra_path):
    # Location of excel data#excel数据的位置
    excelData = pd.read_csv(spectra_path, header = 0, index_col = 0)    
    excelDataSpectra = excelData.iloc[:,:800] #index 直到 Excel 文件中光谱的最后一个点
    excelDataTensor = torch.tensor(excelDataSpectra.values).type(torch.FloatTensor)
    return excelData, excelDataSpectra, excelDataTensor

excelData, excelDataSpectra, excelDataTensor = Excel_Tensor(spectra_path)

f = open('training_log.txt','w')
start_time = time.time()
local_time = time.ctime(start_time)
print('Start Time = %s' % local_time)
print('Start Time = %s' % local_time, file=f)

#Does not truncate tensor contents (Can set "Default")#不截断张量内容（可以设置“默认”）
torch.set_printoptions(profile="full")

#Set random seed for reproducibility#设置随机种子，以实现可复制性
manualSeed = 999
#manualSeed = random.randint(1, 10000)# 如果你想要新的结果，请使用
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

#数据加载器的worker数量（对于Windows工作者必须=0，参考：https://github.com/pytorch/pytorch/issues/2341）。
workers = 1 

#训练期间的批量大小
batch_size = 16

#训练图像的空间大小。所有图像都将使用transformer.调整到这个大小。 
image_size = 64
#训练图像中的通道数。对于彩色图像，这是 3is 3
nc = 3 

 #z 潜在向量的大小（即生成器输入的大小）)
latent = 400
gan_input = excelDataTensor.size()[1] + latent

#Size of feature maps #生成器中特征图的大小generator
ngf = 128

#Size of feature maps in discriminator
#判别器中特征图的大小
ndf = 64

#Number of training epochs#训练epoch数
num_epochs = 1

#Learning rate for optimizers#优化器的学习率
lr = 0.0001

#Beta1 hyperparam for Adam optimizers#Beta1 用于 Adam 优化器的超参数
beta1 = 0.5

#Number of GPUs available. Use 0 for CPU mode.#可用的 GPU 数量。使用 0 表示 CPU 模式。
ngpu = 1

#Create the dataset. Use "dataset.imgs" to show filenames
#创建数据集。使用“dataset.imgs”显示文件名
dataset = dset.ImageFolder(root=img_path,
                           transform=transforms.Compose([
                               transforms.Resize(image_size),
                               transforms.CenterCrop(image_size),
                               transforms.ToTensor(),
                               transforms.Normalize([0.5],[0.5]) 

                               
#Create the dataloader
#创建数据加载器
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=False, num_workers=workers)

#Decide which device we want to run on
#决定我们要在哪个设备上运行                               
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

#Custom weights initialization called on netG and netD
#在netG和netD上调用自定义权重初始化
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

#Generator Code
#生成器代码                              
class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu            
        self.conv1 = nn.ConvTranspose2d(gan_input, ngf * 8, 6, 1, 0, bias=False)
        self.conv2 = nn.BatchNorm2d(ngf * 8)
        self.conv3 = nn.ReLU(True)
        # state size. (ngf*8) x 6 x 6
        self.conv4 = nn.ConvTranspose2d(ngf * 8, ngf * 4, 6, 2, 2, bias=False)
        self.conv5 = nn.BatchNorm2d(ngf * 4)
        self.conv6 = nn.ReLU(True)
        # state s7ze. (ngf*4) x 12 x 12
        self.conv7 = nn.ConvTranspose2d(ngf * 4, ngf * 2, 6, 2, 4, bias=False)
        self.conv8 = nn.BatchNorm2d(ngf * 2)
        self.conv9 = nn.ReLU(True)
        # state size. (ngf*2) x 20 x 20
        self.conv10 = nn.ConvTranspose2d(ngf * 2, ngf, 6, 2, 5, bias=False)
        self.conv11 = nn.BatchNorm2d(ngf)
        self.conv12 = nn.ReLU(True)
        # state size. (ngf) x 36 x 36
        self.conv13 = nn.ConvTranspose2d(ngf, nc, 6, 2, 4, bias=False)
        self.conv14 = nn.Tanh()
        # state size. (nc) x 68 x 68

    def forward(self, input):
        imageOut = input
        imageOut = self.conv1(imageOut)
        imageOut = self.conv2(imageOut)
        imageOut = self.conv3(imageOut)
        imageOut = self.conv4(imageOut)
        imageOut = self.conv5(imageOut)
        imageOut = self.conv6(imageOut)
        imageOut = self.conv7(imageOut)
        imageOut = self.conv8(imageOut)
        imageOut = self.conv9(imageOut)
        imageOut = self.conv10(imageOut)
        imageOut = self.conv11(imageOut)
        imageOut = self.conv12(imageOut)
        imageOut = self.conv13(imageOut)
        imageOut = self.conv14(imageOut)               
        return imageOut

#Create the generator
#创建生成器
netG = Generator(ngpu).to(device)

#Handle multi-gpu if desired
#如果需要，处理多GPU
if (device.type == 'cuda') and (ngpu > 1):
    netG = nn.DataParallel(netG, list(range(ngpu)))

#Apply the weights_init function to randomly initialize all weights to mean=0, stdev=0.2.
#应用weights_init函数将所有权重随机初始化为mean=0，stdev=0.2。
netG.apply(weights_init)

#Print the model
print(netG)

class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.l1 = nn.Linear(800, image_size*image_size*nc, bias=False)           
        self.conv1 = nn.Conv2d(2*nc, ndf, 6, 2, 4, bias=False) 
        self.conv2 = nn.LeakyReLU(0.2, inplace=True)
        # state size. (ndf) x 36 x 36
        self.conv3 = nn.Conv2d(ndf, ndf * 2, 6, 2, 5, bias=False)
        self.conv4 = nn.BatchNorm2d(ndf * 2)
        self.conv5 = nn.LeakyReLU(0.2, inplace=True)
        # state size. (ndf*2) x 20 x 20
        self.conv6 = nn.Conv2d(ndf * 2, ndf * 4, 6, 2, 4, bias=False)
        self.conv7 = nn.BatchNorm2d(ndf * 4)
        self.conv8 = nn.LeakyReLU(0.2, inplace=True)
        # state size. (ndf*4) x 12 x 12
        self.conv9 = nn.Conv2d(ndf * 4, ndf * 8, 6, 2, 2, bias=False)
        self.conv10 = nn.BatchNorm2d(ndf * 8)
        self.conv11 = nn.LeakyReLU(0.2, inplace=True)
        # state size. (ndf*8) x 6 x 6
        self.conv12 = nn.Conv2d(ndf * 8, 1, 6, 1, 0, bias=False)
        self.conv13 = nn.Sigmoid()

    def forward(self, input, label):
        x1 = input
        x2 = self.l1(label)
        x2 = x2.reshape(int(b_size/ngpu),nc,image_size,image_size) 
        combine = torch.cat((x1,x2),1)
        combine = self.conv1(combine)
        combine = self.conv2(combine)
        combine = self.conv3(combine)
        combine = self.conv4(combine)
        combine = self.conv5(combine)
        combine = self.conv6(combine)
        combine = self.conv7(combine)
        combine = self.conv8(combine)
        combine = self.conv9(combine)
        combine = self.conv10(combine)
        combine = self.conv11(combine)
        combine = self.conv12(combine)
        combine = self.conv13(combine)
        return combine

#Create the Discriminator#创建判别器
netD = Discriminator(ngpu).to(device)

#Handle multi-gpu if desired#如果需要，处理多GPU
if (device.type == 'cuda') and (ngpu > 1):
    netD = nn.DataParallel(netD, list(range(ngpu)))

#Apply the weights_init function to randomly initialize all weights to mean=0, stdev=0.2.#应用weights_init函数将所有权重随机初始化为mean=0，stdev=0.2。
netD.apply(weights_init)

#Print the model
print(netD)

#Initialize BCELoss function#初始化BCELoss函数
criterion = nn.BCELoss()

#Create batch of latent vectors that we will use to visualize the progression of the generator#创建一批潜在向量，我们将使用它们来可视化生成器的进程
testTensor = torch.Tensor()
for i in range (100):
    fixed_noise1 = torch.cat((excelDataTensor[i*int(np.floor(len(excelDataSpectra)/100))],torch.rand(latent)))
    fixed_noise2 = fixed_noise1.unsqueeze(1).unsqueeze(1).unsqueeze(1)
    fixed_noise = fixed_noise2.permute(1,0,2,3)
    testTensor = torch.cat((testTensor,fixed_noise),0)
testTensor = testTensor.to(device)

#Establish convention for real and fake labels during training#在训练期间建立真假标签的约定
real_label = random.uniform(0.9,1.0)
fake_label = 0

#Setup Adam optimizers for both G and D#为 G 和 D 设置 Adam 优化器
optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

##Training Loop
#Lists to keep track of progress
##训练循环
#用于跟踪进度的列表
img_list = []
G_losses = []
D_losses = []
iters = 0
noise = torch.Tensor()
noise2 = torch.Tensor()
print("Starting Training Loop...")
#For each epoch
x=0
for epoch in range(num_epochs):
    x=0
    # For each batch in the dataloader
    for i, data in enumerate(dataloader, 0):
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))更新 D 网络：最大化 log(D(x)) + log(1 - D(G(z)))
        ###########################
        ## Train with all-real batch 使用全实批次进行训练
        netD.zero_grad()
        # Format batch
        real_cpu = data[0].to(device)
        b_size = real_cpu.size(0)
        label = torch.full((b_size,), real_label, device=device)

        # Generate batch of Spectra,  latent vectors, and Properties   生成一批光谱、潜在向量和属性  
        for j in range(batch_size):
            excelIndex = x*batch_size+j
            try:
                gotdata = excelDataTensor[excelIndex]
            except IndexError:
                break
            tensorA = excelDataTensor[excelIndex].view(1,800)
            noise2 = torch.cat((noise2,tensorA),0)      
            
            tensor1 = torch.cat((excelDataTensor[excelIndex],torch.rand(latent)))
            tensor2 = tensor1.unsqueeze(1).unsqueeze(1).unsqueeze(1)         
            tensor3 = tensor2.permute(1,0,2,3)
            noise = torch.cat((noise,tensor3),0)         
                              
        noise = noise.to(device)            
        noise2 = noise2.to(device)                
        
         # Forward pass real batch through D   通过 D 前向传递真实批次
        output = netD.forward(real_cpu,noise2).view(-1)
        # Calculate loss on all-real batch计算全实批次的损失
        errD_real = criterion(output, label)
        # Calculate gradients for D in backward pass在反向传播中计算 D 的梯度
        errD_real.backward()
        D_x = output.mean().item()
              
        ## Train with all-fake batch     使用全假批次进行训练           
        # Generate fake image batch with G  用 G 生成假图像批次
        fake = netG.forward(noise)
        label.fill_(fake_label)
        # Classify all fake batch with D  用 D 对所有假批次进行分类
        output = netD.forward(fake.detach(),noise2).view(-1)
        # Calculate D's loss on the all-fake batch  计算 D 在全假批次上的损失
        errD_fake = criterion(output, label)
        # Calculate the gradients for this batch  计算这批的梯度
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        # Add the gradients from the all-real and all-fake batches  添加来自全真和全假批次的渐变
        errD = errD_real + errD_fake
        # Update D 更新
        optimizerD.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))  更新 G 网络：最大化 log(D(G(z)))
        ###########################
        netG.zero_grad()
        label.fill_(real_label)  # fake labels are real for generator cost  # 虚假标签对于生成器成本来说是真实的
        # Since we just updated D, perform another forward pass of all-fake batch through D  因为我们刚刚更新了D，所以再执行一次通过D的全假批的前向传递
        output = netD.forward(fake,noise2).view(-1)
        # Calculate G's loss based on this output  根据这个输出计算 G 的损失
        errG = criterion(output, label)
        # Calculate gradients for G  计算 G 的梯度
        errG.backward()
        D_G_z2 = output.mean().item()
        # Update G# 更新 G
        optimizerG.step()

        # Output training stats# 输出训练数据
        if i % 50 == 0:
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                  % (epoch, num_epochs, i, len(dataloader),
                     errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                  % (epoch, num_epochs, i, len(dataloader),
                     errD.item(), errG.item(), D_x, D_G_z1, D_G_z2), file=f)

        # Save Losses for plotting later# 保存损失以供以后绘制
        G_losses.append(errG.item())
        D_losses.append(errD.item())

       #  Check how the generator is doing by saving G's output on fixed_noise通过将 G 的输出保存在 fixed_noise 上来检查生成器的运行情况
        if (iters % 500 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
            with torch.no_grad():
                fake = netG(testTensor).detach().cpu()
            img_list.append(vutils.make_grid(fake, nrow=10, padding=2, normalize=True))

        iters += 1
        noise = torch.Tensor()
        noise2 = torch.Tensor()     
        x += 1
    if epoch % 50 == 0:
        ##Update folder location
        torch.save(netG, save_dir + 'netG' + str(epoch) + '.pt')
        torch.save(netD, save_dir + 'netD' + str(epoch) + '.pt')

local_time = time.ctime(time.time())
print('End Time = %s' % local_time)
print('End Time = %s' % local_time, file=f)
run_time = (time.time()-start_time)/3600
print('Total Time Lapsed = %s Hours' % run_time)
print('Total Time Lapsed = %s Hours' % run_time, file=f)
f.close()


#Save training progress video#保存训练进度视频
ims, ani = Video.save_video(save_dir, img_list, G_losses, D_losses)


#Plot and save G and D Training Losses#绘制并保存 G 和 D 训练损失
plt.figure(figsize=(10,5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(G_losses,label="Generator Loss")
plt.plot(D_losses,label="Discriminator Loss")
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.legend()
plt.savefig('losses.png')
plt.show()

