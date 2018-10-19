# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 11:14:09 2017

@author: Administrator
"""

import numpy as np
import torch.nn as nn
import torch.nn.parallel
import argparse
import os
import time
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import scipy.io as sio
# import matplotlib.pyplot as plt
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.utils as vutils
from sklearn.metrics import confusion_matrix
import torchvision.models as models
import utils


parser = argparse.ArgumentParser()
parser.add_argument('--workers', type=int, help='number of data loading workers', default=0)
parser.add_argument('--dataset_name', default='Salinas', help='dataset name')
parser.add_argument('--batch_size', type=int, default=20, help='input batch size')
parser.add_argument('--patch_size', type=int, default=8, help='the height / width of the input image to network')
parser.add_argument('--count', type=int, default=5, help='number of training samples selected from each class')
parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
parser.add_argument('--nch', type=int, default=3)
parser.add_argument('--ngf', type=int, default=256)
parser.add_argument('--ndf', type=int, default=256)
parser.add_argument('--n_epochs', type=int, default=600, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--manualSeed', type=int, default=2018, help='manual seed')
parser.add_argument('--decreasing_lr', default='30,60,90,150,200,300,400,500', help='decreasing strategy')
parser.add_argument('--wd', type=float, default=0.0005, help='weight decay')
parser.add_argument('--clamp_lower', type=float, default=-0.01)
parser.add_argument('--clamp_upper', type=float, default=0.01)
opt = parser.parse_args()


batch_size = opt.batch_size
patch_size = opt.patch_size
nz = opt.nz
ngf = opt.ngf
ndf = opt.ndf
n_epochs = opt.n_epochs
ncls = 0
nch = 0
rng = np.random.RandomState(opt.manualSeed)
torch.manual_seed(2018)
torch.cuda.manual_seed_all(2018)
dataset_name = opt.dataset_name
count = opt.count

cudnn.benchmark = True


def load_data():
    import cv2
    from zca import ZCA
    zca = ZCA()

    # load dataset
    dataset = np.load('dataset/{}.npy'.format(dataset_name)).item()
    data = dataset['data']
    data_map = dataset['label']
    global nch, ncls
    nch = data.shape[2]
    ncls = len(np.unique(data_map)) - 1

    # partition the training and test data
    train_coord = np.empty((0, 2)).astype(np.int8) # coordinates of the training data
    test_coord = np.empty((0, 2)).astype(np.int8) # coordinates of the test data
    for cls in range(ncls):
        coord_class = np.transpose(np.nonzero(data_map == cls + 1))
        rng.shuffle(coord_class)
        # count = int(np.round(len(coord_class) * percent))
        samples_per_class = count
        train_coord = np.concatenate((train_coord, coord_class[:samples_per_class]))
        test_coord = np.concatenate((test_coord, coord_class[samples_per_class:]))
    rng.shuffle(train_coord)
    rng.shuffle(test_coord)
    print(train_coord.shape, test_coord.shape)
    train_map = np.zeros_like(data_map)
    test_map = np.zeros_like(data_map)
    for i in range(train_coord.shape[0]):
        train_map[train_coord[i, 0], train_coord[i, 1]] = data_map[train_coord[i, 0], train_coord[i, 1]]
    for i in range(test_coord.shape[0]):
        test_map[test_coord[i, 0], test_coord[i, 1]] = data_map[test_coord[i, 0], test_coord[i, 1]]

    # data preprocessin
    data = ((data - np.min(data[train_map != 0])) / np.max(
        data[train_map != 0] - np.min(data[train_map != 0])) - 0.5) * 2
    zca.fit(data[train_map != 0])
    data = zca.transform(data.reshape(-1, nch)).reshape(data.shape[0], data.shape[1], data.shape[2])

    # padding the HSI scene and the label map
    data = cv2.copyMakeBorder(data,
                              patch_size // 2,
                              patch_size // 2,
                              patch_size // 2,
                              patch_size // 2,
                              cv2.BORDER_REPLICATE)
    train_map = cv2.copyMakeBorder(train_map,
                                   patch_size // 2,
                                   patch_size // 2,
                                   patch_size // 2,
                                   patch_size // 2,
                                   cv2.BORDER_REPLICATE)
    test_map = cv2.copyMakeBorder(test_map,
                                  patch_size // 2,
                                  patch_size // 2,
                                  patch_size // 2,
                                  patch_size // 2,
                                  cv2.BORDER_REPLICATE)

    train_coord += patch_size // 2
    test_coord += patch_size // 2
    return data, train_map, train_coord, test_map, test_coord


def get_batch(data, samples_map, samples_coord):
    data_size = data.shape
    batch = np.zeros((0, patch_size, patch_size, data_size[2]))
    label = np.zeros((0)).astype(np.int)
    for i in range(samples_coord.shape[0]):
        batch = np.concatenate((batch, np.expand_dims(
            data[(samples_coord[i, 0] - patch_size // 2):(samples_coord[i, 0] + patch_size // 2),
            (samples_coord[i, 1] - patch_size // 2):(samples_coord[i, 1] + patch_size // 2), :], axis=0)))
        label = np.concatenate((label, np.array([samples_map[samples_coord[i, 0], samples_coord[i, 1]]])))
    batch = np.expand_dims(batch, axis=1)
    label = np.squeeze(np.eye(ncls + 1)[(label - 1).reshape(-1)])
    return batch, label


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class generator(nn.Module):
    def __init__(self, nz, nch):
        super(generator, self).__init__()
        init_size = patch_size // 8
        self.LeakyReLU = nn.LeakyReLU(0.2, inplace=True)
        self.Tanh = nn.Tanh()
        self.linear1 = nn.Linear(nz, ngf * 2)
        self.linear2 = nn.Linear(ngf * 2, ngf * 4 * init_size * init_size)

        self.conv1 = nn.ConvTranspose3d(1, 8, (4, 4, 1), (2, 2, 1), (1, 1, 0), bias=False)
        self.BatchNorm1 = nn.BatchNorm3d(8)
        self.conv2 = nn.Conv3d(8, ngf * 2, (1, 1, ngf * 4), 1, 0, bias=False)
        self.BatchNorm2 = nn.BatchNorm3d(ngf * 2)

        self.conv3 = nn.ConvTranspose3d(1, 8, (4, 4, 1), (2, 2, 1), (1, 1, 0), bias=False)
        self.BatchNorm3 = nn.BatchNorm3d(8)
        self.conv4 = nn.Conv3d(8, ngf, (1, 1, ngf * 2), 1, 0, bias=False)
        self.BatchNorm4 = nn.BatchNorm3d(ngf)

        self.conv5 = nn.ConvTranspose3d(1, 8, (4, 4, 1), (2, 2, 1), (1, 1, 0), bias=False)
        self.BatchNorm5 = nn.BatchNorm3d(8)
        self.conv6 = nn.Conv3d(8, nch, (1, 1, ngf), 1, 0, bias=False)

        self.apply(weights_init)

    def forward(self, input):
        init_size = patch_size // 8
        x = self.linear1(input)
        x = self.LeakyReLU(x)
        x = self.linear2(x)
        x = self.LeakyReLU(x)
        x = x.view(x.shape[0], 1, init_size, init_size, ngf * 4)

        x = self.conv1(x)
        x = self.LeakyReLU(x)
        x = self.BatchNorm1(x)
        x = self.conv2(x)
        x = self.LeakyReLU(x)
        x = self.BatchNorm2(x)
        x = x.permute(0 ,4, 2, 3, 1)


        x = self.conv3(x)
        x = self.LeakyReLU(x)
        x = self.BatchNorm3(x)
        x = self.conv4(x)
        x = self.LeakyReLU(x)
        x = self.BatchNorm4(x)
        x = x.permute(0, 4, 2, 3, 1)

        x = self.conv5(x)
        x = self.LeakyReLU(x)
        x = self.BatchNorm5(x)
        x = self.conv6(x)
        x = self.Tanh(x)
        output = x.permute(0 ,4, 2, 3, 1)

        return output


class discriminator(nn.Module):
    def __init__(self, nch, ncls):
        super(discriminator, self).__init__()
        self.LeakyReLU = nn.LeakyReLU(0.2, inplace=True)

        self.conv1 = nn.Conv3d(1, 8, (4, 4, 1), (2, 2, 1), (1, 1, 0), bias=False)
        self.conv2 = nn.Conv3d(8, ndf, (1, 1, nch), 1, 0, bias=False)
        self.BatchNorm2 = nn.BatchNorm3d(ndf)
        self.Drop2 = nn.Dropout3d(p=0.3)

        self.conv3 = nn.Conv3d(1, 8, (4, 4, 1), (2, 2, 1), (1, 1, 0), bias=False)
        self.BatchNorm3 = nn.BatchNorm3d(8)
        self.conv4 = nn.Conv3d(8, ndf * 2, (1, 1, ndf), 1, 0, bias=False)
        self.BatchNorm4 = nn.BatchNorm3d(ndf * 2)
        self.Drop4 = nn.Dropout3d(p=0.3)

        self.conv5 = nn.Conv3d(1, 8, (4, 4, 1), (2, 2, 1), (1, 1, 0), bias=False)
        self.BatchNorm5 = nn.BatchNorm3d(8)
        self.conv6 = nn.Conv3d(8, ndf * 4, (1, 1, ndf * 2), 1, 0, bias=False)

        self.global_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.disc_linear = nn.Linear(ndf * 4, 1)
        self.aux_linear = nn.Linear(ndf * 4, ncls + 1)
        # self.softmax = nn.LogSoftmax()
        # self.sigmoid = nn.Sigmoid()
        self.apply(weights_init)

    def forward(self, input):
        x = self.conv1(input)
        x = self.LeakyReLU(x)
        x = self.conv2(x)
        x = self.LeakyReLU(x)
        x = self.Drop2(x)
        x = self.BatchNorm2(x)
        x = x.permute(0, 4, 2, 3, 1)

        x = self.conv3(x)
        x = self.LeakyReLU(x)
        x = self.BatchNorm3(x)
        x = self.conv4(x)
        x = self.LeakyReLU(x)
        x = self.Drop4(x)
        x = self.BatchNorm4(x)
        x = x.permute(0, 4, 2, 3, 1)

        x = self.conv5(x)
        x = self.LeakyReLU(x)
        x = self.BatchNorm5(x)
        x = self.conv6(x)
        x = self.LeakyReLU(x)

        x = self.global_pool(x).squeeze()
        # print(x.shape)
        c = self.aux_linear(x)
        # c = self.softmax(c)
        # s = self.disc_linear(x)
        # s = self.sigmoid(s)
        return x, c


def D_loss(y_pred, y_true):
    cat_loss = torch.mean(F.nll_loss(F.log_softmax(y_pred, 1), y_true.max(1)[1])\
               * torch.eq(torch.sum(y_true[:, :-1], 1), 1).float())
    real_class_logits, fake_class_logits = torch.split(y_pred, ncls, 1)
    mx = torch.max(real_class_logits, 1, keepdim=True)[0]
    stable_real_class_logits = real_class_logits - mx
    gan_logits = torch.log(torch.sum(torch.exp(stable_real_class_logits), 1, keepdim=True)) + mx - fake_class_logits
    gan_loss_labels = torch.unsqueeze(torch.eq(torch.sum(y_true[:, :-1], 1), 1).float(), -1)
    gan_loss = torch.mean(F.binary_cross_entropy(F.sigmoid(gan_logits), gan_loss_labels))
    return gan_loss + cat_loss


def G_loss(y_pred, y_true):
    return torch.mean(F.mse_loss(y_pred, y_true))


def test(predict, labels):
    pred = predict.data.max(1)[1]
    correct = pred.eq(labels.data).cpu().sum()
    return correct, len(labels.data)


def build_model():
    netG = generator(nz, nch)

        # if opt.netG != '':
        #     netG.load_state_dict(torch.load(opt.netG))
        # print(netG)

    netD = discriminator(nch, ncls)

        # if opt.netD != '':
        #     netD.load_state_dict(torch.load(opt.netD))
        # print(netD)
    return netD, netG


def train_model(netD, netG, data, train_map, train_coord):
    s_criterion = nn.BCELoss()
    c_criterion = nn.NLLLoss()

    input_lab = torch.FloatTensor(batch_size, 1, patch_size, patch_size, nch)
    input_unl = torch.FloatTensor(batch_size, 1, patch_size, patch_size, nch)
    input_unl2 = torch.FloatTensor(3 * batch_size, 1, patch_size, patch_size, nch)
    noise_D = torch.FloatTensor(batch_size, nz)
    noise_G = torch.FloatTensor(3 * batch_size, nz)
    # fixed_noise = torch.FloatTensor(batch_size, nz).normal_(0, 1)
    # s_label = torch.FloatTensor(batch_size)
    # c_label = torch.LongTensor(batch_size)
    label = torch.LongTensor(batch_size, ncls + 1)

    netD.cuda()
    netG.cuda()
    # s_criterion.cuda()
    # c_criterion.cuda()
    input_lab = input_lab.cuda()
    input_unl = input_unl.cuda()
    input_unl2 = input_unl2.cuda()
    # s_label = s_label.cuda()
    # c_label = c_label.cuda()
    label = label.cuda()
    noise_D = noise_D.cuda()
    noise_G = noise_G.cuda()
    # fixed_noise = fixed_noise.cuda()

    input_lab = Variable(input_lab)
    input_unl = Variable(input_unl)
    input_unl2 = Variable(input_unl2)
    # s_label = Variable(s_label)
    # c_label = Variable(c_label)
    label = Variable(label)
    noise_D = Variable(noise_D)
    noise_G = Variable(noise_G)
    # fixed_noise = Variable(fixed_noise)
    optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, weight_decay=opt.wd)
    optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, weight_decay=opt.wd)
    decreasing_lr = list(map(int, opt.decreasing_lr.split(',')))
    print('decreasing_lr: ' + str(decreasing_lr))

    best_acc = 0
    t_begin = time.time()
    for epoch in range(1, n_epochs + 1):
        netD.train()
        netG.train()
        right = 0
        if epoch in decreasing_lr:
            optimizerD.param_groups[0]['lr'] *= 0.6
            optimizerG.param_groups[0]['lr'] *= 0.6

        train_coord_lab = train_coord[np.random.permutation(train_coord.shape[0])]
        train_coord_unl = train_coord[np.random.permutation(train_coord.shape[0])]
        train_coord_unl2 = train_coord[np.random.permutation(train_coord.shape[0])]
        n_steps = train_coord_lab.shape[0] // batch_size
        discr_loss_lab = 0
        discr_loss_unl = 0
        discr_loss_fake = 0
        gen_loss = 0
        for step in range(n_steps):
            # train with labeled
            netD.zero_grad()
            data_lab_batch, label_batch = get_batch(data, train_map,
                                                    train_coord_lab[step * batch_size:(step + 1) * batch_size])
            input_lab.data.copy_(torch.from_numpy(data_lab_batch))
            label.data.copy_(torch.from_numpy(label_batch))
            _, logits = netD(input_lab)
            discr_loss_lab = D_loss(logits, label)
            # discr_loss_lab.backward()
            # D_x = s_output.data.mean()
            correct, length = test(logits, label.max(1)[1])
            right += correct

            # train with unlabeled
            data_unl_batch, data_unl_label = get_batch(data, train_map, train_coord_unl[step * batch_size:(step + 1) * batch_size])
            input_unl.data.copy_(torch.from_numpy(data_unl_batch))
            label.data.copy_(torch.zeros(data_unl_label.shape))
            _, logits = netD(input_lab)
            discr_loss_unl = D_loss(logits, label)
            # discr_loss_unl.backward()

            # train with fake

            noise_D.data.resize_(batch_size, nz)
            noise_D.data.normal_(0, 1)
            noise_ = np.random.normal(0, 1, (batch_size, nz))
            noise_ = (torch.from_numpy(noise_)).float()
            noise_ = noise_.resize_(batch_size, nz)
            noise_D.data.copy_(noise_)

            label_batch = np.zeros((batch_size, ncls + 1))
            label_batch[:, -1] = 1

            label.data.copy_(torch.from_numpy(label_batch))

            fake_D = netG(noise_D)
            _, logits = netD(fake_D.detach())
            discr_loss_fake = D_loss(logits, label)
            # discr_loss_fake.backward()
            # D_G_z1 = s_output.data.mean()
            # errD = errD_real.data[0] + errD_fake.data[0]
            discr_loss = discr_loss_lab + discr_loss_unl + discr_loss_fake
            discr_loss.backward()
            optimizerD.step()

            ###############
            #  Updata G
            ##############
            netG.zero_grad()
            data_unl2_batch, _ = get_batch(data, train_map,
                train_coord_unl2[rng.permutation(train_coord.shape[0])][:3 * batch_size])
            # s_label.data.fill_(real_label)  # fake labels are real for generator cost
            noise_G.data.resize_(3 * batch_size, nz)
            noise_G.data.normal_(0, 1)
            noise_ = np.random.normal(0, 1, (3 * batch_size, nz))
            noise_ = (torch.from_numpy(noise_)).float()
            noise_ = noise_.resize_(3 * batch_size, nz)
            noise_G.data.copy_(noise_)
            fake_G = netG(noise_G)
            input_unl2.data.copy_(torch.from_numpy(data_unl2_batch))
            feat_real, _ = netD(input_unl2)
            feat_fake, _ = netD(fake_G)
            gen_loss = G_loss(feat_fake, feat_real)
            gen_loss.backward()
            # D_G_z2 = s_output.data.mean()
            optimizerG.step()


        # if epoch % 10 == 0:
        print('[%d/%d] discr_loss_lab: %.4f, discr_loss_unl: %.4f discr_loss_fake: %.4f, gen_loss: %.4f, Accuracy: %d / %d = %.4f'
              % (epoch, n_epochs, discr_loss_lab, discr_loss_unl, discr_loss_fake, gen_loss, right, n_steps * batch_size, 100. * right / (n_steps * batch_size)))

        # if epoch % 10 == 0:
            # eval_model(data, test_map, test_coord)
            # torch.save(netG.state_dict(), '%s/netG_epoch_%d.pth' % (opt.outf, epoch))
            # torch.save(netD.state_dict(), '%s/netD_epoch_%d.pth' % (opt.outf, epoch))


def eval_model(data, test_map, test_coord):
    netD.eval()
    netG.eval()
    test_loss = 0
    right = 0
    predict = np.array([], dtype=np.int64)
    labels = np.array([], dtype=np.int64)

    n_steps = test_coord.shape[0] // batch_size
    for step in range(n_steps):
        coord_batch = test_coord[step * batch_size: (step + 1) * batch_size]
        x, y = get_batch(data, test_map, coord_batch)
        batch = torch.from_numpy(x).float()
        label = torch.from_numpy(y[:, :-1]).long()
        indx_target = label.max(1)[1].clone()
        batch, label = batch.cuda(), label.cuda()
        batch, label = Variable(batch, volatile=True), Variable(label)
        _, logits = netD(batch)
        pred = F.log_softmax(logits[:, :-1], 1).data.max(1)[1]  # get the index of the max log-probability
        right += pred.cpu().eq(indx_target).sum()
        predict = np.append(predict, pred.cpu().numpy())
        labels = np.append(labels, y[:, :-1].argmax(1))

    acc = 100. * right / (n_steps * batch_size)
    # print('\tTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
    #     test_loss, right, n_steps * batch_size, acc))
    print('Acc:', acc)
    print(labels.shape, predict.shape)
    confuse_mat = confusion_matrix(labels, predict, labels=np.arange(ncls))
    OA = np.trace(confuse_mat) / confuse_mat.sum()
    PA = np.diag(confuse_mat) / confuse_mat.sum(axis=1)
    AA = PA.sum() / (ncls)
    EA = (confuse_mat.sum(axis=1) * confuse_mat.sum(axis=0)).sum() / (confuse_mat.sum()) ** 2
    Kappa = (OA - EA) / (1 - EA)
    print('\tOA = {:.5f}% AA = {:.5f}% kappa * 100 = {:.5f}'.format(OA * 100, AA * 100, Kappa * 100))
    # print('Class accuracy: ', PA * 100)



if __name__ == '__main__':
    data, train_map, train_coord, test_map, test_coord = load_data()
    train_start = time.time()
    netD, netG = build_model()
    train_model(netD, netG, data, train_map, train_coord)
    train_end = time.time()
    eval_model(data, test_map, test_coord)
    eval_end = time.time()
    print('Training time: ', train_end - train_start)
    # print('Evaluation time: ', eval_end - train_end)
