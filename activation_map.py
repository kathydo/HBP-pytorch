#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Fine-tune all layers only for HBP(Hierarchical Bilinear Pooling for Fine-Grained Visual Recognition).
Usage:
    CUDA_VISIBLE_DEVICES=0,1 python HBP_all.py --base_lr 0.001 --batch_size 24 --epochs 200 --weight_decay 0.0005 --model 'HBP_fc_epoch_*.pth' | tee 'hbp_all.log'
"""


import os
import torch
import torchvision
import cub200
#import visdom
import argparse

from skimage import transform

#import os
import copy
#import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms

from PIL import Image
from torch.autograd import Variable
import matplotlib.pyplot as plt
#from CaffeLoader import loadCaffemodel
#from tensorboardX import SummaryWriter
#vis = visdom.Visdom(env=u'HBP_all',use_incoming_socket=False)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)


#Below is a module == neural net layer
#It inputs and outputs layers
class HBP(torch.nn.Module):
    def __init__(self):
        """Declare all needed layers."""
        torch.nn.Module.__init__(self)
        # Convolution and pooling layers of VGG-16.
        self.features = torchvision.models.vgg16(pretrained=False).features
        self.features_conv5_1 = torch.nn.Sequential(*list(self.features.children())
                                            [:-5])  
        self.features_conv5_2 = torch.nn.Sequential(*list(self.features.children())
                                            [-5:-3])  
        self.features_conv5_3 = torch.nn.Sequential(*list(self.features.children())
                                            [-3:-1])     
        self.bilinear_proj = torch.nn.Sequential(torch.nn.Conv2d(512,8192,kernel_size=1,bias=False),
                                        torch.nn.BatchNorm2d(8192),
                                        torch.nn.ReLU(inplace=True))
        # Linear classifier.
        self.fc = torch.nn.Linear(8192*3, 200)

    def hbp(self,conv1,conv2):
        N = conv1.size()[0]
        proj_1 = self.bilinear_proj(conv1)
        proj_2 = self.bilinear_proj(conv2)
        assert(proj_1.size() == (N,8192,28,28))
        X = proj_1 * proj_2
        assert(X.size() == (N,8192,28,28))    
        X = torch.sum(X.view(X.size()[0],X.size()[1],-1),dim = 2)
        X = X.view(N, 8192)   
        X = torch.sqrt(X + 1e-5)
        X = torch.nn.functional.normalize(X)
        return X

    def forward(self, X):
        N = X.size()[0]
        assert X.size() == (N, 3, 448, 448)

        X_conv5_1 = self.features_conv5_1(X)
        # print("X_conv5_1 info")
        # print(X_conv5_1.shape)
        # print(type(X_conv5_1))

        min_5_1 = torch.min(X_conv5_1)
        max_5_1 = torch.max(X_conv5_1)
        # print("Min of entire X_conv_51")
        # print(min_5_1)
        # print("Max of entire X_conv_51")
        # print(min_5_1)


        X_conv5_2 = self.features_conv5_2(X_conv5_1)
        print("X_conv5_2 info")
        print(X_conv5_2.shape)
        print(type(X_conv5_2))

        # Image2PIL = transforms.ToPILImage()
        print("TAKING MEAN ACROSS ALL CHANNELS")
        #resize = torch.nn.Upsample(size=(448, 448), mode='linear')
        #activation_map = resize(X_conv5_1)
        activation_map = torch.mean(X_conv5_1, dim = 1)
        #print(activation_map.shape)
        #print(activation_map)
        #print("min and max of average")

        min_t = torch.min(activation_map)
        max_t = torch.max(activation_map)
        #print(torch.min(activation_map))
        #print(torch.max(activation_map))

        #norm = torch.nn.functional.normalize(average)
        #normal = torchvision.transforms.Normalize(mean=(0, 0, 0), std=(1, 1, 1) )
        #norm = normal(average)
        
        norm = 255 * (activation_map - min_t) / (max_t - min_t)
        # print("Normalized output:")
        # print(norm)
        # print(torch.min(norm))
        # print(torch.max(norm))

        print("********Work for X conv5_3*********")

        X_conv5_3 = self.features_conv5_3(X_conv5_2)
        print("X_conv5_3 shape")
        print(X_conv5_3.shape)

        min_5_3 = torch.min(X_conv5_3)
        max_5_3 = torch.max(X_conv5_3)
        print("min 5_3:")
        print(min_5_3)
        print("max 5_3:")
        print(max_5_3)

        conv5_3_map = torch.mean(X_conv5_3, dim = 1)

        print("After MEAN: X_conv5_3 shape")
        print(conv5_3_map.shape)

        mean_min_5_3 = torch.min(conv5_3_map)
        mean_max_5_3 = torch.max(conv5_3_map)
        print("mean_min 5_3:")
        print(mean_min_5_3)
        print("mean_max 5_3:")
        print(mean_max_5_3)

        #std = torch.std()

        conv5_3_norm = (conv5_3_map - mean_min_5_3) / (mean_max_5_3 - mean_min_5_3)
        torch.clamp(conv5_3_norm, min=0, max=255)
        print(conv5_3_norm)

        print("After Normalization")
        norm_min_5_3 = torch.min(conv5_3_norm)
        norm_max_5_3 = torch.max(conv5_3_norm)

        print("NORM min 5_3:")
        print(norm_min_5_3)
        print("NORM max 5_3:")
        print(norm_max_5_3)


        '''
        print("conv5_3_map")
        print(conv5_3_map)

        with torch.no_grad():
            conv5_3_map_ng = torch.mean(X_conv5_3, dim = 1)

        print("conv5_3 map no gradient")
        print(conv5_3_map_ng)
        '''

        #resize = torch.nn.Upsample(size=(448, 448), mode='bilinear')
        #image = resize(average)
        
        Image2PIL = transforms.ToPILImage()
        image = Image2PIL(conv5_3_norm)
        img = transforms.Resize(size = 448)(image)
        img.save(str('conv5_3_norm_resize.jpg'))

        # min_t52 = torch.min(X_conv5_2)
        # max_t52 = torch.max(X_conv5_2)
        # print(torch.min(X_conv5_2))
        # print(torch.max(X_conv5_2))

        #conv52_norm = 255 * (X_conv5_2 - min_t52) / (max_t52 - min_t52)

        # image2_mean = torch.mean(conv52_norm, dim = 1)

        # image2 = Image2PIL(image2_mean)
        # image2.save(str('conv5_2.jpg'))

        

        #output_tensor = torch.Tensor(3, 448, 448)

        #deprocess(X_conv5_1, 448, 'birdytest.jpg')
        # print(X_conv5_1)
        # X_conv5_2 = self.features_conv5_2(X_conv5_1)
        # X_conv5_3 = self.features_conv5_3(X_conv5_2)
        
        # X_branch_1 = self.hbp(X_conv5_1,X_conv5_2)
        # X_branch_2 = self.hbp(X_conv5_2,X_conv5_3)
        # X_branch_3 = self.hbp(X_conv5_1,X_conv5_3)

        # X_branch = torch.cat([X_branch_1,X_branch_2,X_branch_3],dim=1)
        # assert X_branch.size() == (N,8192*3)

        # X = self.fc(X_branch)
        # assert X.size() == (N, 200)
        return X

class HBPManager(object):
    def __init__(self, path):
        print('Prepare the network and data.')
        #self._options = options
        self._path = path
        # Network.
        #self._net = torch.nn.DataParallel(HBP()).cuda()
        #self._net = HBP()
        #print(self._net)
        hbp_load = torch.load(path, map_location='cpu')
        self._net = HBP()
        #load_model = torch.load(path, map_location='cpu')
        self._net.load_state_dict(torch.load(path, map_location='cpu'), strict = False)
        

        print("Printing parameters")  
        #print(self._net.parameters())     

        #for param in self._net.parameters():
        #    print(param)

        '''
        print('load_model')
        #print(load_model)
        print("attributes of object")
        print(dir(load_model))
        print("load_model.keys")
        print(load_model.keys)
        print("See here!!!!!")
        #print(load_model.items) 
        print(type(load_model))
        #print(load_model.items())
        #print(dir(load_model.items()))
        #print(load_model.items['bilinear_proj'])
        '''


        #hbp_load = self._net.load_state_dict(torch.load(path, map_location='cpu'), strict = False)

        print("Successfully loaded model")

def preprocess(image_name, image_size):
    image = Image.open(image_name).convert('RGB')
    train_transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize(size=448),  # Let smaller edge match
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.RandomCrop(size=448),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                             std=(0.229, 0.224, 0.225))
        ])
    cropped = train_transforms(image) 

    print("cropped type") #tensor
    print(type(cropped))
    print("cropped.shape")
    print(cropped.shape)

    cropped.data.numpy()

    imgplot = plt.imshow(cropped)
    plt.show(imgplot)

    # cropped_PIL = transforms.ToPILImage(cropped)
    # cropped_PIL.save("cropped.jpg")
    #image_size = tuple([int((float(image_size) / max(image.size))*x) for x in (image.height, image.width)]) 
    # Loader = transforms.Compose([transforms.Resize(image_size), transforms.ToTensor()])  # resize and convert to tensor
    # rgb2bgr = transforms.Compose([transforms.Lambda(lambda x: x[torch.LongTensor([2,1,0])]) ])
    # Normalize = transforms.Compose([transforms.Normalize(mean=[103.939, 116.779, 123.68], std=[1,1,1]) ]) # Subtract BGR
    # tensor = Variable(Normalize(rgb2bgr(Loader(image) * 256))).unsqueeze(0)
    tensor = Variable(cropped).unsqueeze(0)
    Image2PIL = transforms.ToPILImage()
    cropped_PIL = Image2PIL(tensor)
    cropped_PIL.save("cropped.jpg")
    return tensor.float(), image_size
 
# Undo the above preprocessing and save the tensor as an image:
def deprocess(output_tensor, image_size, output_name):
    #Normalize = transforms.Compose([transforms.Normalize(mean=[-103.939, -116.779, -123.68], std=[1,1,1]) ]) # Add BGR
    #bgr2rgb = transforms.Compose([transforms.Lambda(lambda x: x[torch.LongTensor([2,1,0])]) ])
    #ResizeImage = transforms.Compose([transforms.Resize(image_size)])
    #output_tensor = bgr2rgb(Normalize(output_tensor.squeeze(0))) / 256
    output_tensor.clamp_(0, 1)
    print("output tensor info")
    print(output_tensor.size)
    print(type(output_tensor))
    print(output_tensor)
    Image2PIL = transforms.ToPILImage()
    image = Image2PIL(output_tensor)
    image = ResizeImage(image)
    image.save(str(output_name))


def main():
    
    image_path = 'Laysan_Albatross_0056_500.jpg'
    model_path = '/Users/kathydo/Documents/GitHub/pytorch-convis-master/models/HBP_all_epoch_223.pth'

    im = Image.open(image_path)
    
    train_transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize(size=448),  # Let smaller edge match
            #torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.CenterCrop(size=448),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                             std=(0.229, 0.224, 0.225))
        ])
    cropped = train_transforms(im)
    # print("im.size")
    # print(im.size)
    # print("im")
    # print(im)
    # Image2PIL = transforms.ToPILImage()
    # image = Image2PIL(cropped)
    # image.save(str('cropped_bird.jpg'))

    #print("cropped.size")
    #print(cropped.shape)

    tensor = Variable(cropped).unsqueeze(0)
    #print("tensor shape")
    #print(tensor.shape)

    #hbp_load = torch.load(model_path, map_location='cpu')
    manager = HBPManager(model_path)
    #hbp_forward = hbp_load._net(tensor)
    forward = manager._net(tensor)

    print("got here")
    #manager._net(tensor)

if __name__ == '__main__':
    main()