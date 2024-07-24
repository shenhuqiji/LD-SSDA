from datetime import datetime
import os
import os.path as osp
import timeit
from torchvision.utils import make_grid
import time
import numpy as np
import pytz
import torch
import torch.nn.functional as F

from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from utils.IOU import iou_mean

import socket
from utils.metrics import *
from utils.Utils import *


mseloss = torch.nn.MSELoss()

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

class Trainer(object):

    def __init__(self, cuda, model_gen, model_domain_dis, model_uncertainty_dis, optimizer_gen, optimizer_domain_dis,optimizer_uncertainty_dis,
                 domain_loaderU_val,domain_loaderU, domain_loaderS, domain_loaderT, out, max_epoch, stop_epoch=None,
                 lr_gen=1e-3, lr_dis=1e-3, lr_decrease_rate=0.1, interval_validate=None, batch_size=8):
        self.cuda = cuda
        self.model_gen = model_gen
        self.model_dis2 = model_uncertainty_dis
        self.model_dis = model_domain_dis
        self.optim_gen = optimizer_gen
        self.optim_dis = optimizer_domain_dis
        self.optim_dis2 = optimizer_uncertainty_dis
        self.lr_gen = lr_gen
        self.lr_dis = lr_dis
        self.lr_decrease_rate = lr_decrease_rate
        self.batch_size = batch_size

        self.domain_loaderU_val = domain_loaderU_val
        self.domain_loaderU = domain_loaderU
        self.domain_loaderS = domain_loaderS
        self.domain_loaderT = domain_loaderT
        self.time_zone = 'Asia/Hong_Kong'
        self.timestamp_start = \
            datetime.now(pytz.timezone(self.time_zone))

        self.interval_validate = interval_validate

        self.out = out

        log_dir = os.path.join(self.out, 'tensorboard',
                               datetime.now().strftime('%b%d_%H-%M-%S') + '_' + socket.gethostname())
        self.writer = SummaryWriter(log_dir=log_dir)

        self.epoch = 0
        self.iteration = 0
        self.max_epoch = max_epoch
        self.stop_epoch = stop_epoch if stop_epoch is not None else max_epoch
        self.best_disc_dice = 0.0
        self.running_loss_tr = 0.0
        self.running_adv_diff_loss = 0.0
        self.running_adv_same_loss = 0.0
        self.best_mean_dice = 0.0
        self.best_epoch = -1


    def train_epoch(self):
        source_domain_label = 1
        target_domain_label = 0
        same_domain_label = 1
        differ_domain_label = 0
        smooth = 1e-7
        # self.model_gen.train()
        # self.model_dis.train()
        # self.model_dis2.train()
        self.running_seg_acc = 0.0
        self.running_seg_loss = 0.0
        self.running_seg_tar_loss = 0.0
        self.running_adv_loss = 0.0
        self.running_dis_diff_loss = 0.0
        self.running_dis_same_loss = 0.0
        self.running_total_loss = 0.0
        self.running_cup_dice_tr = 0.0
        self.running_disc_dice_tr = 0.0

        domain_u_loader = enumerate(self.domain_loaderU)
        domain_t_loader = enumerate(self.domain_loaderT)
        start_time = timeit.default_timer()
        for batch_idx, sampleS in enumerate(self.domain_loaderS):

            self.optim_gen.zero_grad()
            self.optim_dis.zero_grad()
            self.optim_dis2.zero_grad()

            # 1. train generator with random images
            for param in self.model_dis.parameters():
                param.requires_grad = False
            for param in self.model_dis2.parameters():
                param.requires_grad = False
            for param in self.model_gen.parameters():
                param.requires_grad = True

            ### source image
            imageS = sampleS['image'].cuda()
            labelS_map = sampleS['label'].cuda()
            oS, latent_oS = self.model_gen(imageS)

            ### labeled target image
            try:
                _, sampleT = next(domain_t_loader)
            except:
                domain_t_loader = enumerate(self.domain_loaderT)
                _, sampleT = next(domain_t_loader)
            imageT = sampleT['image'].cuda()
            labelT_map = sampleT['label'].cuda()
            oT, latent_oT = self.model_gen(imageT)

            weights = [0.4, 1]
            class_weights = torch.FloatTensor(weights).cuda()
            celoss = CrossEntropyLoss(weight=class_weights)#torch.nn.BCELoss()

            loss_seg_S1 = celoss(oS, labelS_map)
            loss_seg_T1 = celoss(oT, labelT_map)
            acc_S = iou_mean(torch.argmax(oS, dim=1), labelS_map)
            acc_T = iou_mean(torch.argmax(oT, dim=1), labelT_map)

            acc_avg = (acc_S + acc_T)/2
            loss_seg = loss_seg_S1 + loss_seg_T1

            self.running_seg_acc += acc_avg
            self.running_seg_loss += loss_seg.item()

            loss_seg.backward(retain_graph=True)

            # # 2. train generator with images from different domain
            try:
                id_, sampleU = next(domain_u_loader)
            except:
                domain_u_loader = enumerate(self.domain_loaderU)
                id_, sampleU = next(domain_u_loader)

            imageU = sampleU['image'].cuda()

            oU, latent_oU = self.model_gen(imageU)

            #### unlabeled target loss ###
            # weights_p = [0, 1]
            # weights_n = [0, 0.4]
            # class_weights_p = torch.FloatTensor(weights_p).cuda()
            # class_weights_n = torch.FloatTensor(weights_n).cuda()
            # pce_loss = CrossEntropyLoss(weight=class_weights_p)
            # nce_loss = CrossEntropyLoss(weight=class_weights_n)
            # positive_learning = torch.argmax(F.softmax(oU,dim=1),dim=1)
            # negative_learning = torch.argmin(F.softmax(oU,dim=1),dim=1)
            # Feature0, Feature1 = F.softmax(oU,dim=1).split(1,dim=1)
            # positive_learning[torch.squeeze(Feature1, dim=1) < 0.7]=0
            # negative_learning[torch.squeeze(Feature0, dim=1) < 0.7]=0
            # ones = torch.FloatTensor(oU.data.size()).fill_(1).cuda()
            # loss_target_bce = -1.0 * torch.sum(positive_learning * torch.log(Feature1) + 0.5 * negative_learning * torch.log(Feature0)) / (oU.size(0) * oU.size(2) * oU.size(3))
            # # loss_target_bce = pce_loss(F.softmax(oU,dim=1),positive_learning) + nce_loss(ones-F.softmax(oU,dim=1),negative_learning)
            # # loss_target_bce = nce_loss(ones-F.softmax(oU,dim=1),negative_learning)
            # self.running_seg_tar_loss += loss_target_bce.item()
            # loss_target_bce.backward(retain_graph=True)

            ### Adv loss ####
            uncertainty_mapU = -1.0 * torch.sigmoid(oU) * torch.log(torch.sigmoid(oU) + smooth)
            D_out1 = self.model_dis2(uncertainty_mapU)

            Domain_same= self.model_dis(F.sigmoid(latent_oS), F.sigmoid(latent_oU)) ### latent_space
            # Domain_same= self.model_dis(F.sigmoid(oS), F.sigmoid(oU)) ### output_space


            loss_adv_diff1 = F.binary_cross_entropy_with_logits(D_out1, torch.FloatTensor(D_out1.data.size()).fill_(source_domain_label).cuda())
            loss_adv_diff2 = F.binary_cross_entropy_with_logits(Domain_same, torch.FloatTensor(Domain_same.data.size()).fill_(same_domain_label).cuda())
            loss_adv_diff = loss_adv_diff1 + loss_adv_diff2

            self.running_adv_diff_loss += loss_adv_diff.item()

            loss_adv_diff.backward()

            print('epoch: %d , iteration: %d/%d , loss_S : %f, loss_T : %f' %
                  (self.epoch, len(self.domain_loaderS), batch_idx, loss_seg_S1.item(), loss_seg_T1.item()))

            # 3. train discriminator with images from same domain
            for param in self.model_dis.parameters():
                param.requires_grad = True
            for param in self.model_dis2.parameters():
                param.requires_grad = True
            for param in self.model_gen.parameters():
                param.requires_grad = False

            oS = oS.detach()
            oT = oT.detach()
            oU = oU.detach()
            latent_oS = latent_oS.detach()
            latent_oT = latent_oT.detach()
            latent_oU = latent_oU.detach()

            uncertainty_mapS = -1.0 * torch.softmax(oS,dim=1) * torch.log(torch.softmax(oS,dim=1) + smooth)
            D_outS = self.model_dis2(uncertainty_mapS)

            Domain_same = self.model_dis(F.sigmoid(latent_oT),F.sigmoid(latent_oU)) ### latent_space
            # Domain_same = self.model_dis(F.sigmoid(oT),F.sigmoid(oU)) ### output_space

            loss_D_same1 = F.binary_cross_entropy_with_logits(D_outS, torch.FloatTensor(D_outS.data.size()).fill_(
                source_domain_label).cuda())
            loss_D_same2 = F.binary_cross_entropy_with_logits(Domain_same, torch.FloatTensor(Domain_same.data.size()).fill_(
                same_domain_label).cuda())

            loss_D_same = loss_D_same1 + loss_D_same2

            self.running_dis_same_loss += loss_D_same.item()
            loss_D_same.backward()

            # 4. train discriminator with images from different domain

            uncertainty_mapU = -1.0 * torch.softmax(oU,dim=1) * torch.log(torch.softmax(oU,dim=1) + smooth)
            D_outT = self.model_dis2(uncertainty_mapU)

            Domain_differ = self.model_dis(F.sigmoid(latent_oS),F.sigmoid(latent_oU)) ### latent_space
            # Domain_differ = self.model_dis(F.sigmoid(oS),F.sigmoid(oU)) ### output_space

            loss_D_diff1 = F.binary_cross_entropy_with_logits(D_outT, torch.FloatTensor(D_outT.data.size()).fill_(
                target_domain_label).cuda())
            loss_D_diff2 = F.binary_cross_entropy_with_logits(Domain_differ, torch.FloatTensor(Domain_differ.data.size()).fill_(
                differ_domain_label).cuda())
            loss_D_diff = loss_D_diff1 + loss_D_diff2

            self.running_dis_diff_loss += loss_D_diff.item()
            loss_D_diff.backward()

            # 5. update parameters
            self.optim_gen.step()
            self.optim_dis.step()
            self.optim_dis2.step()

            print('epoch: %d , iteration: %d/%d ,  loss_adv_diff : %f, loss_D_same : %f, loss_D_diff: %f' %
            (self.epoch, len(self.domain_loaderU),id_, loss_adv_diff.item(), loss_D_same.item(), loss_D_diff.item()))

        self.running_seg_acc /= len(self.domain_loaderS)
        self.running_seg_loss /= len(self.domain_loaderS)
        # self.running_seg_tar_loss /= len(self.domain_loaderS)
        self.running_adv_diff_loss /= len(self.domain_loaderS)
        self.running_dis_same_loss /= len(self.domain_loaderS)
        self.running_dis_diff_loss /= len(self.domain_loaderS)

        stop_time = timeit.default_timer()

        torch.save({'model_state_dict': self.model_gen.state_dict()}, osp.join(self.out, 'checkpoint_%d.pth.tar' % self.epoch))
        # torch.save({'model_D1_state_dict': self.model_dis.state_dict()}, osp.join(self.out, 'checkpoint_D1_%d.pth.tar' % self.epoch))
        # torch.save({'model_D2_state_dict': self.model_dis2.state_dict()}, osp.join(self.out, 'checkpoint_D2_%d.pth.tar' % self.epoch))


        print('\n[Epoch: %d] lr:%f,  segLoss: %f,  seg_acc: %f '
              'advLoss: %f, dis_same_Loss: %f, '
              'dis_diff_Lyoss: %f,'
              'Execution time: %.5f' %
              (self.epoch, get_lr(self.optim_gen), self.running_seg_loss, self.running_seg_acc,
               self.running_adv_diff_loss, self.running_dis_same_loss,
               self.running_dis_diff_loss, stop_time - start_time))

        line = ('epoch %d : loss_seg: %f, loss_acc: %f'
                % (self.epoch, self.running_seg_loss, self.running_seg_acc))
        with open('logs_train.txt', 'a') as f:
            f.write(line + '\n')

    def validate(self):

        self.running_seg_acc_val = 0.0
        self.running_seg_loss_val = 0.0

        training = self.model_gen.training
        self.model_gen.eval()

        with torch.no_grad():
            for batch_idx, sample_val in enumerate(self.domain_loaderU_val):
                ### val image
                image_val = sample_val['image'].cuda()
                label_val = sample_val['label'].cuda()
                o_val, _ = self.model_gen(image_val)

                weights = [0.4, 1]
                class_weights = torch.FloatTensor(weights).cuda()
                celoss = CrossEntropyLoss(weight=class_weights)#torch.nn.BCELoss()

                loss_seg_val = celoss(o_val, label_val)
                acc_val = iou_mean(torch.argmax(o_val, dim=1), label_val)

                self.running_seg_acc_val += acc_val
                self.running_seg_loss_val += loss_seg_val.item()

            self.running_seg_acc_val /= len(self.domain_loaderU_val)
            self.running_seg_loss_val /= len(self.domain_loaderU_val)


            line = ('epoch %d : loss_seg: %f, loss_acc: %f'
                    % (self.epoch, self.running_seg_acc_val, self.running_seg_loss_val))
            print(line)
            with open('logs_test.txt', 'a') as f:
                f.write(line + '\n')

            if training:
                self.model_gen.train()
                self.model_dis.train()
                self.model_dis2.train()

    def train(self):
        for epoch in range(1, self.max_epoch+1):
            self.epoch = epoch
            self.train_epoch()
            if self.stop_epoch == self.epoch:
                print('Stop epoch at %d' % self.stop_epoch)
                break

            if (epoch) % 20 == 0:
                self.lr_gen = self.lr_gen * 0.1
                for param_group in self.optim_gen.param_groups:
                    param_group['lr'] = self.lr_gen
            self.writer.add_scalar('lr_gen', get_lr(self.optim_gen), self.epoch * (len(self.domain_loaderS)))
            if self.epoch % self.interval_validate == 0:
                self.validate()
        self.writer.close()



