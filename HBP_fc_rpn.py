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
import argparse
import numpy as np

from tensorboardX import SummaryWriter

from torch.autograd import Variable

from roi_align.crop_and_resize import CropAndResizeFunction

torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

def generate_anchors(base_size=16, ratios=[0.5, 1, 2],
                     scales=2**np.arange(3, 6)):

    sub_sample = 16
    ratio = [0.5, 1, 2]
    anchor_scales = [4, 8, 16]

    fe_size = (448//16) #28

    #crearting centers from first location at feature map location (1,1)
    ctr_x = np.arange(32, (fe_size) * 16, 16)
    ctr_y = np.arange(32, (fe_size) * 16, 16)

    '''
    Now ctr x is take off first and last layer of pixels
    ctr_x.shape
    (26,)
    [ 32  48  64  80  96 112 128 144 160 176 192 208 224 240 256 272 288 304
 320 336 352 368 384 400 416 432]
    '''
    #There are 26 * 26 locations from feature map
    ctr = np.zeros((26 * 26, 2))

    #creating centre coordinates
    index = 0
    for x in range(len(ctr_x)):
        for y in range(len(ctr_y)):
            ctr[index, 1] = ctr_x[x] - 8
            ctr[index, 0] = ctr_y[y] - 8
            index +=1

    #creating anchor array
    anchors = np.zeros((26 * 26 * 9, 4))
    index = 0
    
    #generate all anchor locations
    for c in ctr:
      ctr_y, ctr_x = c
      for i in range(len(ratios)):
        for j in range(len(anchor_scales)):
          h = sub_sample * anchor_scales[j] * np.sqrt(ratios[i])
          w = sub_sample * anchor_scales[j] * np.sqrt(1./ ratios[i])
          
          anchors[index, 0] = ctr_y - h / 2.
          anchors[index, 1] = ctr_x - w / 2.
          anchors[index, 2] = ctr_y + h / 2.
          anchors[index, 3] = ctr_x + w / 2.
          index += 1

    #get indexes of invalid boxes
    invalid_index = np.where(
        (anchors[:, 0] < 0) |
        (anchors[:, 1] < 0) |
        (anchors[:, 2] > 448) |
        (anchors[:, 3] > 448)
    )[0]

    #set all invalid anchors to 0
    anchors[invalid_index] = [0,0,0,0]

    #scale anchors coordinates to in range (0,1)
    scaled_anchors = anchors / 448

    tensor_anchors = torch.tensor(scaled_anchors, dtype=torch.float)

    return tensor_anchors

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

        self.bilinear_proj = torch.nn.Sequential(torch.nn.Conv2d(512,1024,kernel_size=1,bias=False),
                                        torch.nn.BatchNorm2d(1024),
                                        torch.nn.ReLU(inplace=True))
        
        #DEFINE RPN
        self.crop_height = 7
        self.crop_width = 7
        self.anchors = generate_anchors()
        self.n_anchor = 9 # Number of anchors at each location
        self.rpn_conv = torch.nn.Conv2d(512, 512, kernel_size = 3,  stride = 1, padding = 0)
        
        self.rpn_cls_layer = torch.nn.Conv2d(512, 9, kernel_size = 1, stride = 1, padding = 0)

        self.sigmoid = torch.nn.Sigmoid()

        # init conv sliding layer
        self.rpn_conv.weight.data.normal_(0, 0.01)
        self.rpn_conv.bias.data.zero_()

        # init classification layer
        self.rpn_cls_layer.weight.data.normal_(0, 0.01)
        self.rpn_cls_layer.bias.data.zero_()

        # Linear classifier.
        self.fc = torch.nn.Linear(1024*3, 200)

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

        #get interactions between projections
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

        anchors = self.anchors

        #get tensor of all anchors coordinates for N samples
        anchors = torch.cat(([anchors] * N),0)

        boxes_data = torch.FloatTensor(anchors)

        #going through code to integrate HBP_anchors to here.
        #use some pytorch operation like expand or repeat to 
        #get box_index concattenated

        box_index_data = torch.tensor(range(N), dtype=torch.int32)
        box_index_data = box_index_data.repeat(6084,1)
        box_index_data = box_index_data.transpose(0,1)
        box_index_data = box_index_data.reshape(N*6084)

        boxes = Variable(boxes_data, requires_grad=False)
        box_index = Variable(box_index_data, requires_grad=False)

        X_conv4_3 = self.features_conv4_3(X)

        X_conv4_3_down = self.resize_halve(X_conv4_3)

        X_conv5_1 = self.features_conv5_1(X)
        X_conv5_2 = self.features_conv5_2(X_conv5_1)
        X_conv5_3 = self.features_conv5_3(X_conv5_2)

        X_conv4_add_5_1 = X_conv5_1.add(X_conv4_3_down)
        X_conv4_add_5_2 = X_conv5_2.add(X_conv4_3_down)
        X_conv4_add_5_3 = X_conv5_3.add(X_conv4_3_down)

        #Set above feature maps to requires_grad = False for cropping

        X_fmap_1 = Variable(X_conv4_add_5_1, requires_grad=False)
        X_fmap_2 = Variable(X_conv4_add_5_2, requires_grad=False)
        X_fmap_3 = Variable(X_conv4_add_5_3, requires_grad=False)

        #Cropping feature maps to 7x7 boxes
        X_fmap_1_cropped = CropAndResizeFunction(self.crop_height, self.crop_width, 0)(X_fmap_1, boxes, box_index)
        X_fmap_2_cropped = CropAndResizeFunction(self.crop_height, self.crop_width, 0)(X_fmap_2, boxes, box_index)
        X_fmap_3_cropped = CropAndResizeFunction(self.crop_height, self.crop_width, 0)(X_fmap_3, boxes, box_index)
        
        #reshape to match rpn output
        X_fmap_1_reshape = np.reshape(X_fmap_1_cropped, (N, 9, 26, 26, 512,7,7))
        X_fmap_2_reshape = np.reshape(X_fmap_2_cropped, (N, 9, 26, 26, 512,7,7))
        X_fmap_3_reshape = np.reshape(X_fmap_3_cropped, (N, 9, 26, 26, 512,7,7))

        #Apply RPN
        rpn_conv1 = self.rpn_conv(X_conv4_add_5_1)
        rpn_conv2 = self.rpn_conv(X_conv4_add_5_2)
        rpn_conv3 = self.rpn_conv(X_conv4_add_5_3)

        cls_scores_1 = self.rpn_cls_layer(rpn_conv1)
        cls_scores_2 = self.rpn_cls_layer(rpn_conv1)
        cls_scores_3 = self.rpn_cls_layer(rpn_conv1)

        rpn_scores_1 = self.sigmoid(cls_scores_1)
        rpn_scores_2 = self.sigmoid(cls_scores_2)
        rpn_scores_3 = self.sigmoid(cls_scores_3)

        #is there a better way using view()?
        rpn_scores_reshape_1 = rpn_scores_1.view(N, 9, 26, 26, 1, 1, 1)
        rpn_scores_reshape_2 = rpn_scores_2.view(N, 9, 26, 26, 1, 1, 1)
        rpn_scores_reshape_3 = rpn_scores_3.view(N, 9, 26, 26, 1, 1, 1)

        rpn_sum_1 = torch.sum(rpn_scores_1)
        rpn_sum_2 = torch.sum(rpn_scores_2)
        rpn_sum_3 = torch.sum(rpn_scores_3)

        weighted_sc_num_1 = rpn_scores_reshape_1 * X_fmap_1_reshape
        weighted_sc_num_2 = rpn_scores_reshape_2 * X_fmap_2_reshape
        weighted_sc_num_3 = rpn_scores_reshape_3 * X_fmap_3_reshape

        sum_num_1 = torch.sum(weighted_sc_num_1, dim = (1,2,3))
        sum_num_2 = torch.sum(weighted_sc_num_2, dim = (1,2,3))
        sum_num_3 = torch.sum(weighted_sc_num_3, dim = (1,2,3))

        weighted_score_1 = sum_num_1/rpn_sum_1
        weighted_score_2 = sum_num_2/rpn_sum_2
        weighted_score_3 = sum_num_3/rpn_sum_3

        X_branch_1 = self.hbp(weighted_score_1, weighted_score_2)
        X_branch_2 = self.hbp(weighted_score_2, weighted_score_3)
        X_branch_3 = self.hbp(weighted_score_1, weighted_score_3)

        X_branch = torch.cat([X_branch_1,X_branch_2,X_branch_3],dim = 1)
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

        tr_writer = SummaryWriter(
        log_dir=os.path.join(self._options['log_dir'], "train"))
        va_writer = SummaryWriter(
        log_dir=os.path.join(self._options['log_dir'], "valid"))

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

            #num_correct = torch.tensor(num_correct).float().cuda()
            num_correct = num_correct.clone().detach().float().cuda()
            #num_correct = num_correct.clone().detach().requires_grad_(True).float()

            num_total = torch.tensor(num_total).float().cuda()
            #num_total = num_total.clone().detach().float().cuda()

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
                                        'HBP_fc_1024_epoch_%d.pth' % (t + 1)))
            

            tr_writer.add_scalar('Training Loss', sum(epoch_loss) / len(epoch_loss), t + 1)
            tr_writer.add_scalar('Training Accuracy', train_acc, t + 1)
            va_writer.add_scalar('Validation Accuracy', test_acc, t + 1)
            
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
        #num_correct = torch.tensor(num_correct).float().cuda()
        num_correct = num_correct.clone().detach().float().cuda()
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

    '''

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
    '''

    base_lr = 1.0
    batch_size = 12
    epochs = 120
    weight_decay = 0.000005
    log_dir = 'logs_fc_rpn'

    options = {
        'base_lr': base_lr,
        'batch_size': batch_size,
        'epochs': epochs,
        'weight_decay': weight_decay,
        'log_dir': log_dir,
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
