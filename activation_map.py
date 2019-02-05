#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Fine-tune the fc layer only for  HBP(Hierarchical Bilinear Pooling for Fine-Grained Visual Recognition).
Usage:
    CUDA_VISIBLE_DEVICES=0,1 python HBP_fc.py --base_lr 1.0 --batch_size 12 --epochs 120 --weight_decay 0.000005 | tee 'hbp_fc.log'
"""

import os
import torch
import torchvision
import cub200
#import visdom
import argparse
from tensorboardX import SummaryWriter

#vis = visdom.Visdom(env=u'HBP_fc',use_incoming_socket=False)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

class HBP(torch.nn.Module):
    def __init__(self):
        torch.nn.Module.__init__(self)
        # Convolution and pooling layers of VGG-16.
        self.features = torchvision.models.vgg16(pretrained=True).features

        #print('VGG16 Pre trained all features:')
        #print(torch.nn.Sequential(*list(self.features.children())[20:]) )

        #getting conv4_3 by taking layers before maxpool
        self.features_conv4_3 = torch.nn.Sequential(*list(self.features.children())
                                            [:23]) 

        #print("Conv 4_3. whatis it?")
        #print(self.features_conv4_3)

        self.features_conv5_1 = torch.nn.Sequential(*list(self.features.children())
                                            [:-5]) 
        #self.conv5_resize = torch.nn.Upsample(scale_factor = 2, mode='bilinear')

        self.features_conv5_2 = torch.nn.Sequential(*list(self.features.children())
                                            [-5:-3])  
        self.features_conv5_3 = torch.nn.Sequential(*list(self.features.children())
                                            [-3:-1])     
        #self.bilinear_proj = torch.nn.Sequential(torch.nn.Conv2d(512,8192,kernel_size=1,bias=False),
        self.bilinear_proj = torch.nn.Sequential(torch.nn.Conv2d(1024,8192,kernel_size=1,bias=False),
                                        torch.nn.BatchNorm2d(8192),
                                        torch.nn.ReLU(inplace=True))
        # Linear classifier.
        self.fc = torch.nn.Linear(8192*3, 200)

        # Freeze all previous layers.
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

        assert(proj_1.size() == (N,8192,56,56))
        #assert(proj_1.size() == (N,8192,28,28))    //old dimension 28*28
        
        X = proj_1 * proj_2

        assert(X.size() == (N,8192,56,56))
        #assert(X.size() == (N,8192,28,28))    
        
        X = torch.sum(X.view(X.size()[0],X.size()[1],-1),dim = 2)
        X = X.view(N, 8192)     
        X = torch.sqrt(X + 1e-5)
        X = torch.nn.functional.normalize(X)
        return X

    def forward(self, X):
        N = X.size()[0]     #N = 12

        #print("What is N????????????????/")
        #print(N)

        assert X.size() == (N, 3, 448, 448)
        
        resize_double = torch.nn.Upsample(scale_factor = 2, mode='bilinear')

        X_conv4_3 = self.features_conv4_3(X)

        #print("X_conv4_3 size")
        #print(X_conv4_3.shape)

        
        X_conv5_1 = self.features_conv5_1(X)
        X_conv5_2 = self.features_conv5_2(X_conv5_1)
        X_conv5_3 = self.features_conv5_3(X_conv5_2)

        # print("X_conv5_1 size")
        # print(X_conv5_1.shape)

        X_conv5_1_resize = resize_double(X_conv5_1)
        X_conv5_2_resize = resize_double(X_conv5_2)
        X_conv5_3_resize = resize_double(X_conv5_3)

        #print("X_conv5_1_resize size")
        # print(X_conv5_1_resize.shape)

        #not sure about dimension , set to 0 for now
        X_conv4_concat5_1 = torch.cat((X_conv4_3, X_conv5_1_resize), 1)
        X_conv4_concat5_2 = torch.cat((X_conv4_3, X_conv5_2_resize), 1)
        X_conv4_concat5_3 = torch.cat((X_conv4_3, X_conv5_3_resize), 1)

        # print("X_conv4_concat5_1 size")
        # print(X_conv4_concat5_1.shape)
        
        X_branch_1 = self.hbp(X_conv4_concat5_1, X_conv4_concat5_2)
        X_branch_2 = self.hbp(X_conv4_concat5_2, X_conv4_concat5_3)
        X_branch_3 = self.hbp(X_conv4_concat5_1, X_conv4_concat5_3)

        X_branch = torch.cat([X_branch_1,X_branch_2,X_branch_3],dim = 1)
      

        '''
        X_conv5_1 = self.features_conv5_1(X)
        X_conv5_2 = self.features_conv5_2(X_conv5_1)
        X_conv5_3 = self.features_conv5_3(X_conv5_2)
        
        X_branch_1 = self.hbp(X_conv5_1,X_conv5_2)
        X_branch_2 = self.hbp(X_conv5_2,X_conv5_3)
        X_branch_3 = self.hbp(X_conv5_1,X_conv5_3)

        X_branch = torch.cat([X_branch_1,X_branch_2,X_branch_3],dim = 1)
        '''

        # print("What is X_branch.size?????")
        # print(X_branch.size())

        assert X_branch.size() == (N,8192*3)
        X = self.fc(X_branch)
        assert X.size() == (N, 200)
        return X

class HBPManager(object):
    def __init__(self, options, path):

        self._options = options
        self._path = path

        print(self._options)
        print(self._path)
        # Network.
        self._net = torch.nn.DataParallel(HBP()).cuda()
        print(self._net)
        # Criterion.
        self._criterion = torch.nn.CrossEntropyLoss().cuda()
        # Solver.
        param_to_optim = []
        for param in self._net.parameters():
            if param.requires_grad == False:
                continue
            param_to_optim.append(param)

        self._solver = torch.optim.SGD(
            param_to_optim, lr=self._options['base_lr'],
            momentum=0.9, weight_decay=self._options['weight_decay'])

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
        self._train_loader = torch.utils.data.DataLoader(
            train_data, batch_size=self._options['batch_size'],
            shuffle=True, num_workers=4, pin_memory=True)
        self._test_loader = torch.utils.data.DataLoader(
            test_data, batch_size=16,
            shuffle=False, num_workers=4, pin_memory=True)

    def train(self):
        print('Training.')
        best_acc = 0.0
        best_epoch = None
        print('Epoch\tTrain loss\tTrain acc\tTest acc')
        ii = 0

        ## Create summary writer in different sub folders
        
        tr_writer = SummaryWriter(
        log_dir=os.path.join(self._options['log_dir'], "train"))
        va_writer = SummaryWriter(
        log_dir=os.path.join(self._options['log_dir'], "valid"))
        

        #log loss and accuracy to tensorboard
        # Create log directory and save directory if it does not exist
        #if not os.path.exists(self._options['log_dir']):
        #    os.makedirs(self._options['log_dir'])
        # if not os.path.exists(self._options['save_dir']):
        #     os.makedirs(self._options['save_dir'])

        for t in range(self._options['epochs']):
            epoch_loss = []
            num_correct = 0
            num_total = 0
            for X, y in self._train_loader:
                # Data.
                X = torch.autograd.Variable(X.cuda())
                y = torch.autograd.Variable(y.cuda(non_blocking = True))
                # Clear the existing gradients.
                self._solver.zero_grad()
                # Forward pass.
                score = self._net(X)
                loss = self._criterion(score, y)
                epoch_loss.append(loss.item())
                # Prediction.
                _, prediction = torch.max(score.data, 1)
                num_total += y.size(0)
                num_correct += torch.sum(prediction == y.data)
                # Backward pass.
                loss.backward()
                self._solver.step()

                ii += 1
                x = torch.Tensor([ii])
                y = torch.Tensor([loss.item()])
                #vis.line(X=x, Y=y, win='polynomial', update='append' if ii>0 else None)
                tr_writer.add_scalar('X', x, ii)
                tr_writer.add_scalar('Y', y, ii)


            num_correct = torch.tensor(num_correct).float().cuda()
            num_total = torch.tensor(num_total).float().cuda()


            train_acc = 100 * num_correct / num_total
            test_acc = self._accuracy(self._test_loader)
            self._scheduler.step(test_acc)
            if test_acc > best_acc:
                best_acc = test_acc
                best_epoch = t + 1
                print('*', end='')
                # Save model onto disk.
                torch.save(self._net.state_dict(),
                           os.path.join(self._path['model'],
                                        'HBP_fc_epoch_%d.pth' % (t + 1)))
            print('%d\t%4.3f\t\t%4.2f%%\t\t%4.2f%%' %
                  (t+1, sum(epoch_loss) / len(epoch_loss), train_acc, test_acc))
        print('Best at epoch %d, test accuaray %f' % (best_epoch, best_acc))

    def _accuracy(self, data_loader):
        self._net.train(False)
        num_correct = 0
        num_total = 0
        for X, y in data_loader:
            # Data.
            X = torch.autograd.Variable(X.cuda())
            y = torch.autograd.Variable(y.cuda(non_blocking = True))
            # Prediction.
            score = self._net(X)
            _, prediction = torch.max(score.data, 1)
            num_total += y.size(0)
            num_correct += torch.sum(prediction == y.data)
        self._net.train(True)  # Set the model to training phase
        num_correct = torch.tensor(num_correct).float().cuda()
        num_total = torch.tensor(num_total).float().cuda()
        return 100 * num_correct / num_total

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
                        required=True, help='Name of log directory.')
    args = parser.parse_args()
    if args.base_lr <= 0:
        raise AttributeError('--base_lr parameter must >0.')
    if args.batch_size <= 0:
        raise AttributeError('--batch_size parameter must >0.')
    if args.epochs < 0:
        raise AttributeError('--epochs parameter must >=0.')
    if args.weight_decay <= 0:
        raise AttributeError('--weight_decay parameter must >0.')

    options = {
        'base_lr': args.base_lr,
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'weight_decay': args.weight_decay,
        'log_dir': args.log_dir
    }

    project_root = os.popen('pwd').read().strip()
    path = {
        'cub200': '/data/datasets/birds',
        'model': os.path.join(project_root, 'model'),
    }
    for d in path:
        print(path[d])
        assert os.path.isdir(path[d])

    manager = HBPManager(options, path)
    manager.getStat()
    manager.train()

if __name__ == '__main__':
    main()
