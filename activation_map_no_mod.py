'''
Creates activation map overlayed onto cropped input (bird) image
Creates and saves 5 images:
activation_bird_relu5_1.png
activation_bird_relu5_2.png
activation_bird_relu5_3.png
activation_bird_proj_5_1.png
activation_bird_proj_5_2.png
activation_bird_proj_5_3.png
'''


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
import argparse

from skimage import transform

import copy
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms

from PIL import Image
from torch.autograd import Variable
import matplotlib.pyplot as plt

torch.manual_seed(0)
torch.cuda.manual_seed_all(0)


class HBP(torch.nn.Module):
    def __init__(self):
        """Declare all needed layers."""
        torch.nn.Module.__init__(self)
        # Convolution and pooling layers of VGG-16.
        self.features = torchvision.models.vgg16(pretrained=False).features

        #self.features_conv4_3 = torch.nn.Sequential(*list(self.features.children())[:23])
        #self.resize_halve = torch.nn.Upsample(size=(28, 28), mode='bilinear')
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

    def hbp1(self,conv1,conv2):
        N = conv1.size()[0]
        proj_1 = self.bilinear_proj(conv1)
        proj_2 = self.bilinear_proj(conv2)
        assert(proj_1.size() == (N,8192,28,28))
        X = proj_1 * proj_2

        bird = Image.open('cropped_bird.jpg')

        X_map = torch.mean(X, dim = 1)

        mean_min_X = torch.min(X_map)
        mean_max_X = torch.max(X_map)
        print("mean_min_X:")
        print(mean_min_X)
        print("mean_max X:")
        print(mean_max_X)

        X_norm = (X_map - mean_min_X) / (mean_max_X - mean_min_X)

        image = transforms.ToPILImage()(X_norm)
        img = transforms.Resize(size = 448)(image)

        img.save(str('proj_5_1_feature_map.png'))

        img.putalpha(200)

        bird.paste(img, (0, 0), img.convert('RGBA'))
        bird.save('activation_bird_proj_5_1.png')

        assert(X.size() == (N,8192,28,28))    
        X = torch.sum(X.view(X.size()[0],X.size()[1],-1),dim = 2)
        X = X.view(N, 8192)   
        X = torch.sqrt(X + 1e-5)
        X = torch.nn.functional.normalize(X)
        return X

    def hbp2(self,conv1,conv2):
        N = conv1.size()[0]
        proj_1 = self.bilinear_proj(conv1)
        proj_2 = self.bilinear_proj(conv2)
        assert(proj_1.size() == (N,8192,28,28))
        X = proj_1 * proj_2

        bird = Image.open('cropped_bird.jpg')

        X_map = torch.mean(X, dim = 1)

        mean_min_X = torch.min(X_map)
        mean_max_X = torch.max(X_map)
        print("mean_min_X:")
        print(mean_min_X)
        print("mean_max X:")
        print(mean_max_X)

        X_norm = (X_map - mean_min_X) / (mean_max_X - mean_min_X)

        image = transforms.ToPILImage()(X_norm)
        img = transforms.Resize(size = 448)(image)

        img.save(str('proj_5_2_feature_map.png'))

        img.putalpha(200)

        bird.paste(img, (0, 0), img.convert('RGBA'))
        bird.save('activation_bird_proj_5_2.png')

        assert(X.size() == (N,8192,28,28))    
        X = torch.sum(X.view(X.size()[0],X.size()[1],-1),dim = 2)
        X = X.view(N, 8192)   
        X = torch.sqrt(X + 1e-5)
        X = torch.nn.functional.normalize(X)
        return X

    def hbp3(self,conv1,conv2):
        N = conv1.size()[0]
        proj_1 = self.bilinear_proj(conv1)
        proj_2 = self.bilinear_proj(conv2)
        assert(proj_1.size() == (N,8192,28,28))
        X = proj_1 * proj_2

        bird = Image.open('cropped_bird.jpg')

        X_map = torch.mean(X, dim = 1)

        mean_min_X = torch.min(X_map)
        mean_max_X = torch.max(X_map)
        print("mean_min_X:")
        print(mean_min_X)
        print("mean_max X:")
        print(mean_max_X)

        X_norm = (X_map - mean_min_X) / (mean_max_X - mean_min_X)

        image = transforms.ToPILImage()(X_norm)
        img = transforms.Resize(size = 448)(image)

        img.save(str('proj_5_3_feature_map.png'))

        img.putalpha(200)

        bird.paste(img, (0, 0), img.convert('RGBA'))
        bird.save('activation_bird_proj_5_3.png')

        assert(X.size() == (N,8192,28,28))    
        X = torch.sum(X.view(X.size()[0],X.size()[1],-1),dim = 2)
        X = X.view(N, 8192)   
        X = torch.sqrt(X + 1e-5)
        X = torch.nn.functional.normalize(X)
        return X

    def forward(self, X):
        N = X.size()[0]
        assert X.size() == (N, 3, 448, 448)

        bird = Image.open('cropped_bird.jpg')
        Image2PIL = transforms.ToPILImage()

        # X_conv4_3 = self.features_conv4_3(X)

        # X_conv4_3_down = self.resize_halve(X_conv4_3)

        X_conv5_1 = self.features_conv5_1(X)
        X_conv5_2 = self.features_conv5_2(X_conv5_1)
        X_conv5_3 = self.features_conv5_3(X_conv5_2)

        # X_conv4_add_5_1 = X_conv5_1.add(X_conv4_3_down)
        # X_conv4_add_5_2 = X_conv5_2.add(X_conv4_3_down)
        # X_conv4_add_5_3 = X_conv5_3.add(X_conv4_3_down)

        X_branch_1 = self.hbp1(X_conv5_1,X_conv5_2)
        X_branch_2 = self.hbp2(X_conv5_2,X_conv5_3)
        X_branch_3 = self.hbp3(X_conv5_1,X_conv5_3)

        #X_branch = torch.cat([X_branch_1,X_branch_2,X_branch_3],dim=1)

        # min_5_1 = torch.min(X_conv4_add_5_1)
        # max_5_1 = torch.max(X_conv4_add_5_1)
        # print("min 5_1:")
        # print(min_5_1)
        # print("max 5_1:")
        # print(max_5_1)

        conv5_1_map = torch.mean(X_conv5_1, dim = 1)

        mean_min_5_1 = torch.min(conv5_1_map)
        mean_max_5_1 = torch.max(conv5_1_map)
        print("mean_min 5_1:")
        print(mean_min_5_1)
        print("mean_max 5_1:")
        print(mean_max_5_1)

        conv5_1_norm = (conv5_1_map - mean_min_5_1) / (mean_max_5_1 - mean_min_5_1)

        image = Image2PIL(conv5_1_norm)
        img = transforms.Resize(size = 448)(image)

        #img.save(str('conv5_1_feature_map.png'))

        img.putalpha(200)

        bird.paste(img, (0, 0), img.convert('RGBA'))
        bird.save('activation_bird_relu5_1.png')

        bird = Image.open('cropped_bird.jpg')

        # min_5_2 = torch.min(X_conv4_add_5_2)
        # max_5_2 = torch.max(X_conv4_add_5_2)
        # print("min 5_2:")
        # print(min_5_2)
        # print("max 5_2:")
        # print(max_5_2)

        conv5_2_map = torch.mean(X_conv5_2, dim = 1)

        mean_min_5_2 = torch.min(conv5_2_map)
        mean_max_5_2 = torch.max(conv5_2_map)
        print("mean_min 5_2:")
        print(mean_min_5_2)
        print("mean_max 5_2:")
        print(mean_max_5_2)

        conv5_2_norm = (conv5_2_map - mean_min_5_2) / (mean_max_5_2 - mean_min_5_2)

        image = Image2PIL(conv5_2_norm)
        img = transforms.Resize(size = 448)(image)

        img.putalpha(200)

        bird.paste(img, (0, 0), img.convert('RGBA'))
        bird.save('activation_bird_relu5_2.png')

        bird = Image.open('cropped_bird.jpg')

        # min_5_3 = torch.min(X_conv4_add_5_3)
        # max_5_3 = torch.max(X_conv4_add_5_3)
        # print("min 5_3:")
        # print(min_5_3)
        # print("max 5_3:")
        # print(max_5_3)

        conv5_3_map = torch.mean(X_conv5_3, dim = 1)

        mean_min_5_3 = torch.min(conv5_3_map)
        mean_max_5_3 = torch.max(conv5_3_map)
        print("mean_min 5_3:")
        print(mean_min_5_3)
        print("mean_max 5_3:")
        print(mean_max_5_3)

        conv5_3_norm = (conv5_3_map - mean_min_5_3) / (mean_max_5_3 - mean_min_5_3)

        image = Image2PIL(conv5_3_norm)
        img = transforms.Resize(size = 448)(image)

        img.putalpha(200)

        bird.paste(img, (0, 0), img.convert('RGBA'))
        bird.save('activation_bird_relu5_3.png')


        return X


def main():
    
    image_path = '/Users/kathydo/Documents/GitHub/birds/birds_ex_paper/Black_Footed_Albatross_0064_796101.jpg'
    model_path = '/Users/kathydo/Documents/GitHub/HBP-pytorch/model/HBP_all_epoch_223.pth'

    im = Image.open(image_path)
    
    train_transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize(size=448), 
            torchvision.transforms.CenterCrop(size=448),
            torchvision.transforms.ToTensor()
        ])
    cropped = train_transforms(im)

    Image2PIL = transforms.ToPILImage()
    bird = Image2PIL(cropped)
    bird.save(str('cropped_bird.jpg'))

    #normalize image
    cropped = torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                             std=(0.229, 0.224, 0.225))(cropped)
    tensor = Variable(cropped).unsqueeze(0)
    
    model = HBP()

    device = torch.device('cpu')
    state_dict = torch.load(model_path, map_location='cpu')

    # create new OrderedDict that does not contain `module.`
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] # remove `module.`
        new_state_dict[name] = v
    # load params
    model.load_state_dict(new_state_dict)
    model = model.eval()

    model.forward(tensor)

if __name__ == '__main__':
    main()