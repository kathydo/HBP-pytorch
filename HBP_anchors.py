'''

based on the code for activation map visualizations,
try to use roi align to crop the image


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
import numpy as np

from skimage import transform

import copy
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms

from PIL import Image
from torch.autograd import Variable
import matplotlib.pyplot as plt
from roi_align.roi_align import RoIAlign
from roi_align.crop_and_resize import CropAndResizeFunction

from model.rpn.rpn import _RPN

torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

try:
    xrange          # Python 2
except NameError:
    xrange = range  # Python 3

def generate_anchors(base_size=16, ratios=[0.5, 1, 2],
                     scales=2**np.arange(3, 6)):
    """
    Generate anchor (reference) windows by enumerating aspect ratios X
    scales wrt a reference (0, 0, 15, 15) window.
    """
    print("What is scale?")
    print(scales)

    base_anchor = np.array([1, 1, base_size, base_size]) - 1

    print("What is base_anchor?")
    print(base_anchor)

    ratio_anchors = _ratio_enum(base_anchor, ratios)

    print("What is ratio_anchors?")
    print(ratio_anchors)

    anchors = np.vstack([_scale_enum(ratio_anchors[i, :], scales)
                         for i in xrange(ratio_anchors.shape[0])])
    return anchors

def _whctrs(anchor):
    """
    Return width, height, x center, and y center for an anchor (window).
    """

    w = anchor[2] - anchor[0] + 1
    h = anchor[3] - anchor[1] + 1
    x_ctr = anchor[0] + 0.5 * (w - 1)
    y_ctr = anchor[1] + 0.5 * (h - 1)
    return w, h, x_ctr, y_ctr

def _mkanchors(ws, hs, x_ctr, y_ctr):
    """
    Given a vector of widths (ws) and heights (hs) around a center
    (x_ctr, y_ctr), output a set of anchors (windows).
    """

    ws = ws[:, np.newaxis]
    hs = hs[:, np.newaxis]
    anchors = np.hstack((x_ctr - 0.5 * (ws - 1),
                         y_ctr - 0.5 * (hs - 1),
                         x_ctr + 0.5 * (ws - 1),
                         y_ctr + 0.5 * (hs - 1)))
    return anchors

def _ratio_enum(anchor, ratios):
    """
    Enumerate a set of anchors for each aspect ratio wrt an anchor.
    """

    w, h, x_ctr, y_ctr = _whctrs(anchor)
    #above values are: 16 16 7.5 7.5

    size = w * h
    size_ratios = size / ratios
    #size ratios [512. 256. 128.]
    ws = np.round(np.sqrt(size_ratios))
    hs = np.round(ws * ratios)
    anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
    return anchors

def _scale_enum(anchor, scales):
    """
    Enumerate a set of anchors for each scale wrt an anchor.
    """

    w, h, x_ctr, y_ctr = _whctrs(anchor)
    ws = w * scales
    hs = h * scales
    anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
    return anchors


class HBP(torch.nn.Module):
    def __init__(self):
        """Declare all needed layers."""
        torch.nn.Module.__init__(self)
        # Convolution and pooling layers of VGG-16.
        self.features = torchvision.models.vgg16(pretrained=False).features

        self.features_conv4_3 = torch.nn.Sequential(*list(self.features.children())[:23])
        self.resize_halve = torch.nn.Upsample(size=(28, 28), mode='bilinear')
        self.features_conv5_1 = torch.nn.Sequential(*list(self.features.children())
                                            [:-5])  
        self.features_conv5_2 = torch.nn.Sequential(*list(self.features.children())
                                            [-5:-3])  
        self.features_conv5_3 = torch.nn.Sequential(*list(self.features.children())
                                            [-3:-1])     
        self.bilinear_proj = torch.nn.Sequential(torch.nn.Conv2d(512,1024,kernel_size=1,bias=False),
                                        torch.nn.BatchNorm2d(1024),
                                        torch.nn.ReLU(inplace=True))
        # Linear classifier.
        self.fc = torch.nn.Linear(1024*3, 200)

    def hbp(self,conv1,conv2):
        N = conv1.size()[0]

        # bilinear_proj = torch.nn.Sequential(torch.nn.Conv2d(512,7,kernel_size=1,bias=False),
        #                                 torch.nn.BatchNorm2d(7),
        #                                 torch.nn.ReLU(inplace=True))

        print("BEFORE size of proj_1, ie. Conv 1")
        print(conv1.size())
        proj_1 = self.bilinear_proj(conv1)

        print("AFTER size of proj_1")
        print(proj_1.size())
        proj_2 = self.bilinear_proj(conv2)
        #assert(proj_1.size() == (N,1024,28,28))
        #assert(proj_1.size() == (N,1024,7,7))
        X = proj_1 * proj_2
        #Size here is same as proj: (N,1024,7,7)

        #assert(X.size() == (N,1024,28,28))
        #assert(proj_1.size() == (N,1024,7,7))    
        
        X = torch.sum(X.view(X.size()[0],X.size()[1],-1),dim = 2)
        #Size here is flattened to (N, 1024)

        X = X.view(N, 1024)   
        X = torch.sqrt(X + 1e-5)
        X = torch.nn.functional.normalize(X)
        return X

    def forward(self, X):
        N = X.size()[0]
        crop_height = 7
        crop_width = 7
        boxes_data = torch.FloatTensor([[0, 0, 1, 1]])
        box_index_data = torch.IntTensor([0])

        boxes = Variable(boxes_data, requires_grad=False)
        box_index = Variable(box_index_data, requires_grad=False)

        assert X.size() == (N, 3, 448, 448)

        bird = Image.open('cropped_bird.jpg')
        Image2PIL = transforms.ToPILImage()

        X_conv4_3 = self.features_conv4_3(X)

        X_conv4_3_down = self.resize_halve(X_conv4_3)

        X_conv5_1 = self.features_conv5_1(X)
        X_conv5_2 = self.features_conv5_2(X_conv5_1)
        X_conv5_3 = self.features_conv5_3(X_conv5_2)

        X_conv4_add_5_1 = X_conv5_1.add(X_conv4_3_down)
        X_conv4_add_5_2 = X_conv5_2.add(X_conv4_3_down)
        X_conv4_add_5_3 = X_conv5_3.add(X_conv4_3_down)

        X_conv451_torch = Variable(X_conv4_add_5_1, requires_grad=False)

        X_conv451_crop = CropAndResizeFunction(crop_height, crop_width, 0)(X_conv451_torch, boxes, box_index)

        X_conv452_torch = Variable(X_conv4_add_5_2, requires_grad=False)

        X_conv452_crop = CropAndResizeFunction(crop_height, crop_width, 0)(X_conv452_torch, boxes, box_index)

        X_conv453_torch = Variable(X_conv4_add_5_3, requires_grad=False)

        X_conv453_crop = CropAndResizeFunction(crop_height, crop_width, 0)(X_conv4_add_5_3, boxes, box_index)

        X_branch_1 = self.hbp(X_conv451_crop,X_conv452_crop)
        X_branch_2 = self.hbp(X_conv452_crop,X_conv453_crop)
        X_branch_3 = self.hbp(X_conv451_crop,X_conv453_crop)

        # X_branch_1 = self.hbp(X_conv4_add_5_1,X_conv4_add_5_2)
        # X_branch_2 = self.hbp(X_conv4_add_5_2,X_conv4_add_5_3)
        # X_branch_3 = self.hbp(X_conv4_add_5_1,X_conv4_add_5_3)

        X_branch = torch.cat([X_branch_1,X_branch_2,X_branch_3],dim=1)

        # print("X_branch_1.size()")
        # print(X_branch_1.size())

        # print("X_branch.size()")
        # print(X_branch.size())
        assert X_branch.size() == (N,1024*3)

        # crop_height = 7
        # crop_width = 7
        # boxes_data = torch.FloatTensor([[0, 0, 1, 1]])
        # box_index_data = torch.IntTensor([0])

        #X_branch_torch = Variable(X_branch, requires_grad=False)
        # boxes = Variable(boxes_data, requires_grad=False)
        # box_index = Variable(box_index_data, requires_grad=False)

        #crops_torch = CropAndResizeFunction(crop_height, crop_width, 0)(X_branch_torch, boxes, box_index)
        #roi_align = RoIAlign(7, 7)
        #crops = roi_align(X_branch, boxes, box_index)

        print(X_conv453_crop.size())

        print(X_branch_1.size())

        X_branch_1

        X = self.fc(X_branch)
        assert X.size() == (N, 200)

        return X


def main():
    
    image_path = '/Users/kathydo/Documents/GitHub/birds/birds_ex_paper/Black_Footed_Albatross_0064_796101.jpg'
    model_path = 'HBP_all_c45_epoch_142.pth'

    anchors = generate_anchors()

    print("anchors******")
    print(anchors)

    '''
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
    '''

if __name__ == '__main__':
    main()