#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Fine-tune the fc layer only for  HBP(Hierarchical Bilinear Pooling for Fine-Grained Visual Recognition).
Usage:
    CUDA_VISIBLE_DEVICES=0,1 python HBP_fc.py --base_lr 1.0 --batch_size 12 --epochs 120 --weight_decay 0.000005 | tee 'hbp_fc.log'

Kingwood:
python HBP_fc_roialign.py --base_lr 1.0 --batch_size 12 --epochs 120 --weight_decay 0.000005 --log_dir logs_fc_roialign | tee 'hbp_fc_roialign_mar_27.log'


"""
import os
import torch
import torchvision
import cub200
import argparse
import numpy as np

from tensorboardX import SummaryWriter

from torch.autograd import Variable

from pytorch_mask_rcnn.roialign.roi_align.crop_and_resize import CropAndResizeFunction

torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

class HBP(torch.nn.Module):
    def __init__(self):
        torch.nn.Module.__init__(self)
        # Convolution and pooling layers of VGG-16.
        self.features = torchvision.models.vgg16(pretrained=True).features
        self.features_conv4_3 = torch.nn.Sequential(*list(self.features.children())
                                            [:23])
        self.resize_halve = torch.nn.Upsample(size=(28, 28), mode='bilinear')
        self.features_conv5_1 = torch.nn.Sequential(*list(self.features.children())
                                            [:-5])  
        self.features_conv5_2 = torch.nn.Sequential(*list(self.features.children())
                                            [-5:-3])  
        self.features_conv5_3 = torch.nn.Sequential(*list(self.features.children())
                                            [-3:-1])     
        # self.bilinear_proj = torch.nn.Sequential(torch.nn.Conv2d(512,8192,kernel_size=1,bias=False),
        #                                 torch.nn.BatchNorm2d(8192),
        #                                 torch.nn.ReLU(inplace=True))
        self.bilinear_proj = torch.nn.Sequential(torch.nn.Conv2d(512,1024,kernel_size=1,bias=False),
                                        torch.nn.BatchNorm2d(1024),
                                        torch.nn.ReLU(inplace=True))
        # Linear classifier.
        self.fc = torch.nn.Linear(1024*3, 200)

        self.crop_height = 7
        self.crop_width = 7

        # Freeze all previous layers.
        for param in self.features_conv4_3.parameters():
            param.requires_grad = False
        for param in self.features_conv5_1.parameters():
            param.requires_grad = False
        for param in self.features_conv5_2.parameters():
            param.requires_grad = False
        for param in self.features_conv5_3.parameters():
            param.requires_grad = False

        # Initialize the fc layers.    
        torch.nn.init.xavier_normal_(self.fc.weight.data)
        if self.fc.bias is not None:
            torch.nn.init.constant_(self.fc.bias.data, val=0)

        #init
        for m in self.bilinear_proj.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, torch.nn.BatchNorm2d):
                torch.nn.init.constant_(m.weight,1)
                torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_normal_(m.weight)
                torch.nn.init.constant_(m.bias, 0)

    def hbp(self,conv1,conv2):
        N = conv1.size()[0]
        proj_1 = self.bilinear_proj(conv1)
        proj_2 = self.bilinear_proj(conv2)
        assert(proj_1.size() == (N,1024,7,7))

        X = proj_1 * proj_2

        assert(X.size() == (N,1024,7,7))    
        X = torch.sum(X.view(X.size()[0],X.size()[1],-1),dim = 2)
        X = X.view(N, 1024)     
        X = torch.sqrt(X + 1e-5)
        X = torch.nn.functional.normalize(X)
        return X

    def forward(self, X):
        N = X.size()[0]
        assert X.size() == (N, 3, 448, 448)

        box = torch.tensor([[0, 0, 1, 1]], dtype=torch.float64).cuda()
        #box = np.array([[0, 0, 1, 1]])
        '''
        print("box type")
        print(type(box))
        '''
        boxes = torch.tensor(box, dtype=torch.double).cuda()
        boxes_data = torch.cat(([boxes] * N),0)

        # print("box_tensor type")
        # print(type(box_tensor))
        #boxes_data = torch.Tensor([[0, 0, 1, 1]], device='cpu')
        #boxes_data = torch.FloatTensor(box_tensor)
        '''
        print("boxes_data type")
        print(type(boxes_data))
        '''

        box_index_data = torch.zeros(N, dtype=torch.int32)

        boxes = Variable(boxes_data, requires_grad=False).cuda()
        box_index = Variable(box_index_data, requires_grad=False).cuda()

        X_conv4_3 = self.features_conv4_3(X)

        X_conv4_3_down = self.resize_halve(X_conv4_3)

        X_conv5_1 = self.features_conv5_1(X)
        X_conv5_2 = self.features_conv5_2(X_conv5_1)
        X_conv5_3 = self.features_conv5_3(X_conv5_2)

        X_conv4_add_5_1 = X_conv5_1.add(X_conv4_3_down)
        X_conv4_add_5_2 = X_conv5_2.add(X_conv4_3_down)
        X_conv4_add_5_3 = X_conv5_3.add(X_conv4_3_down)

        print("X_conv5_1")
        print(X_conv5_1)


        X_conv451_crop = CropAndResizeFunction(self.crop_height, self.crop_width, 0)(X_conv4_add_5_1, boxes, box_index)

        X_conv452_crop = CropAndResizeFunction(self.crop_height, self.crop_width, 0)(X_conv4_add_5_2, boxes, box_index)

        X_conv453_crop = CropAndResizeFunction(self.crop_height, self.crop_width, 0)(X_conv4_add_5_3, boxes, box_index)
        
        print("X_conv451_crop")
        print(X_conv451_crop)

        X_branch_1 = self.hbp(X_conv451_crop,X_conv452_crop)
        X_branch_2 = self.hbp(X_conv452_crop,X_conv453_crop)
        X_branch_3 = self.hbp(X_conv451_crop,X_conv453_crop)

        # print("X_branch_1")
        # print(X_branch_1)

        X_branch = torch.cat([X_branch_1,X_branch_2,X_branch_3],dim = 1)
        
        #print("X_branch size")
        #print(X_branch.size())
        
        assert X_branch.size() == (N,1024*3)
        X = self.fc(X_branch)
        assert X.size() == (N, 200)
        return X

class HBPManager(object):
    def __init__(self, options, path):
        self._options = options
        self._path = path
        # Network.
        self._net = torch.nn.DataParallel(HBP()).cuda()
        print(self._net)
        # Criterion.
        #self._criterion = torch.nn.CrossEntropyLoss().cuda()
        self._criterion = torch.nn.CrossEntropyLoss().cuda()
        # Solver.
        param_to_optim = []
        for param in self._net.parameters():
            if param.requires_grad == False:
                continue
            param_to_optim.append(param)

        # Using ADAM
        self._solver = torch.optim.Adam(
            param_to_optim, lr=self._options['base_lr'],
            weight_decay=self._options['weight_decay'])

        # Commented 
        # self._solver = torch.optim.SGD(
        #     param_to_optim, lr=self._options['base_lr'],
        #     momentum=0.9, weight_decay=self._options['weight_decay'])

        milestones = [40,60,80,100]
        self._scheduler = torch.optim.lr_scheduler.MultiStepLR(self._solver,milestones = milestones,gamma=0.25)

        train_transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize(size=448),  # Let smaller edge match
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.RandomCrop(size=448),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                             std=(0.229, 0.224, 0.225))
        ])
        test_transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize(size=448),
            torchvision.transforms.CenterCrop(size=448),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                             std=(0.229, 0.224, 0.225))
        ])
        train_data = cub200.CUB200(
            root=self._path['cub200'], train=True, download=True,
            transform=train_transforms)
        test_data = cub200.CUB200(
            root=self._path['cub200'], train=False, download=True,
            transform=test_transforms)

        batch_indices = [536,54,4400,1769,1518,1287,5554,4919,2547,2249,5757,589] #12 indices
        '''
        5508,3672,4026,2859,1701,3799,5087,2493,3256,5808,4979,3464,4302,4653,3412,
        5356,4758,2591,3814,2814,2647,3535,4422,4417,1810,1946,2256,2536,3457,5382,
        4850,3930,306,2658,3066,298,109,4691,2458,4867,5841,1283,640,3646,5635,4700,
        2306,5317,5666,595,4930,4011,906,3592,1372,3972,3415,594,1364,4541,4727,5026,
        1939,2917,347,5931,970,622,5380,2673,1145,1339,3743,1192,1606,1075,5740,2909,
        4918,3315,3741,1486,149,3765,3123,1756,5098]
        '''
        subset_sampler = torch.utils.data.sampler.SubsetRandomSampler(batch_indices)

        self._train_loader = torch.utils.data.DataLoader(
            train_data, batch_size=self._options['batch_size'],
            shuffle=False, sampler=subset_sampler, num_workers=4, pin_memory=True)
            #shuffle=True, num_workers=4, pin_memory=True) #changed to shuffle=false for one batch
        self._test_loader = torch.utils.data.DataLoader(
            test_data, batch_size=self._options['test_batch_size'],
            shuffle=False, num_workers=4, pin_memory=True)

    def train(self):
        #print('Training.')
        best_acc = 0.0
        best_epoch = None
        #print('Epoch\tTrain loss\tTrain acc\tTest acc')
        ii = 0

        '''
        tr_writer = SummaryWriter(
        log_dir=os.path.join(self._options['log_dir'], "train"))
        va_writer = SummaryWriter(
        log_dir=os.path.join(self._options['log_dir'], "valid"))
        '''
        iteration = 0
        for t in range(self._options['epochs']):
            epoch_loss = []
            num_correct = 0
            num_total = 0
            self._net.train()
            self._scheduler.step()
            counter = 0
            for batch_idx, (X, y) in enumerate(self._train_loader):
                counter += 1
                # Data.
                # X = torch.autograd.Variable(X.cuda())
                # y = torch.autograd.Variable(y.cuda(non_blocking = True))
                X, y = X.cuda(), y.cuda()

                # print("X.size")
                # print(X.shape)
                # Clear the existing gradients.
                self._solver.zero_grad()
                # Forward pass.
                score = self._net(X)
                loss = self._criterion(score, y)
                # Backward pass.
                loss.backward()
                self._solver.step()

                iteration += 1
                #Start logging
                if iteration % 12 == 0:
                    # Add the values to Summary Writer
                    # self.writer.add_scalar('train/loss', loss.item(), self.args.iter_count)
                    # train_acc = 100 * num_correct / num_total
                    
                # epoch_loss.append(loss.item())

                    # Prediction for current batch
                    _, prediction = torch.max(score.data, 1)
                    num_total = y.size(0)
                    num_correct = torch.sum(prediction == y.data)

                    train_acc = 100 * num_correct / num_total
                    # print("train_acc:", train_acc.item())

                    print('Train Epoch: {} Iteration: {} Processed: [{}/{} ({:.0f}%)]\tLoss_tr: {:.6f} \tTrain_Acc: {:.2f} '.format(
                        t, iteration, batch_idx * len(X), len(self._train_loader.dataset),
                        100. * batch_idx / len(self._train_loader), loss.item(), train_acc.item()))

    def _accuracy(self, data_loader):
        # self._net.train(False) # Abhi
        num_correct = 0
        num_total = 0
        loss_test = 0
        for X, y in data_loader:
            # Data.
            X = torch.autograd.Variable(X.cuda())
            y = torch.autograd.Variable(y.cuda(non_blocking = True))
            # Prediction.
            score = self._net(X)
            _, prediction = torch.max(score.data, 1)
            num_total += y.size(0)
            num_correct += torch.sum(prediction == y.data)
            # Calculate loss
            loss_test += self._criterion(score, y)
        # self._net.train(True)  # Set the model to training phase ---> Abhi
        #num_correct = torch.tensor(num_correct).float().cuda()
        num_correct = num_correct.clone().detach().float().cuda()
        num_total = torch.tensor(num_total).float().cuda()
        return 100 * num_correct / num_total, loss_test//len(data_loader)

    def getStat(self):
        print('Compute mean and variance for training data.')
        train_data = cub200.CUB200(
            root=self._path['cub200'], train=True,
            transform=torchvision.transforms.ToTensor(), download=True)
        train_loader = torch.utils.data.DataLoader(
            train_data, batch_size=1, shuffle=False, num_workers=4,
            pin_memory=True)
        mean = torch.zeros(3)
        std = torch.zeros(3)
        for X, _ in train_loader:
            for d in range(3):
                mean[d] += X[:, d, :, :].mean()
                std[d] += X[:, d, :, :].std()
        mean.div_(len(train_data))
        std.div_(len(train_data))
        print(mean)
        print(std)


def main():

    parser = argparse.ArgumentParser(
        description='Train HBP on CUB200.')
    parser.add_argument('--base_lr', dest='base_lr', type=float, required=True,
                        help='Base learning rate for training.')
    parser.add_argument('--batch_size', dest='batch_size', type=int,
                        required=True, help='Batch size.')
    parser.add_argument('--epochs', dest='epochs', type=int,
                        required=True, help='Epochs for training.')
    parser.add_argument('--weight_decay', dest='weight_decay', type=float,
                        required=True, help='Weight decay.')
    parser.add_argument('--log_dir', dest='log_dir', type=str,
                        required=False, help='Name of log directory.')
    args = parser.parse_args()
    if args.base_lr <= 0:
        raise AttributeError('--base_lr parameter must >0.')
    if args.batch_size <= 0:
        raise AttributeError('--batch_size parameter must >0.')
    if args.epochs < 0:
        raise AttributeError('--epochs parameter must >=0.')
    if args.weight_decay <= 0:
        raise AttributeError('--weight_decay parameter must >0.')

    # base_lr = 3e-3
    # batch_size = 8
    # epochs = 120
    # weight_decay = 0.000005  # Default val: 0.000005
    # log_dir = 'logs_fc_roialign'

    test_batch_size = 1

    options = {
        'base_lr': args.base_lr,
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'weight_decay': args.weight_decay,
        #'log_dir': args.log_dir,
        'test_batch_size' : test_batch_size
    }

    project_root = os.popen('pwd').read().strip()
    path = {
        'cub200': '/data/datasets/birds',
        'model': os.path.join(project_root, 'model'),
    }
    for d in path:
        print(d)
        assert os.path.isdir(path[d])

    manager = HBPManager(options, path)
    manager.getStat()
    manager.train()

if __name__ == '__main__':
    main()
