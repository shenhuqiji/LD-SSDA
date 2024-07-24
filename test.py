#!/usr/bin/env python
from thop import profile
import argparse
import os,time
import os.path as osp
import torch.nn.functional as F

import torch
from torch.autograd import Variable
import tqdm
from dataloaders import lanes_dataloader as DL
from torch.utils.data import DataLoader
from dataloaders import custom_transforms as tr
from torchvision import transforms
from utils.Utils import *
from utils.metrics import *

# from networks.deeplabv3 import *
from networks.erfnet import ERFNet
import cv2, math
import torch.backends.cudnn as cudnn
from networks.HT import hough_transform


cudnn.benchmark = True
cudnn.fastest = True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-file', type=str, default=r'./checkpoint.pth.tar',
                        help='Model path')
    parser.add_argument(
        '--dataset', type=str, default='VIL100', help='test folder id contain images ROIs to test'
    )
    parser.add_argument('-g', '--gpu', type=int, default=0)

    parser.add_argument(
        '--data-dir',
        default=r'./dataset/',
        help='data root path'
    )
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    model_file = args.model_file

    # 1. dataset
    composed_transforms_test = transforms.Compose([
        tr.Normalize_tf(),
        tr.ToTensor()
    ])
    db_test = DL.FundusSegmentation(base_dir=args.data_dir, dataset=args.dataset, split='val',
                                    transform=None, domain='target')

    test_loader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=1)

    # 2. model
    # vote_index = hough_transform(rows=32, cols=64, theta_res=3, rho_res=1)
    # vote_index = torch.from_numpy(vote_index).float().contiguous().cuda()
    # model = ERFNet(num_classes=2,vote_index=vote_index).cuda()
    model = ERFNet(num_classes=2).cuda()

    if torch.cuda.is_available():
        model = model.cuda()
    print('==> Loading %s model file: %s' %
          (model.__class__.__name__, model_file))
    checkpoint = torch.load(model_file)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print('==> Evaluating with %s' % (args.dataset))


    for batch_idx, sample in enumerate(test_loader):
        data = sample['image']
        #target = sample['label']
        img_name = sample['img_name']
        if torch.cuda.is_available():
            data = data.cuda()#, target.cuda()
        data = Variable(data)# Variable(target)
        end = time.time()

        ### 计算复杂度与参数量
        # flops, params = profile(model, inputs=(data,))
        # print(flops, params)

        prediction, _ = model(data)
        output = F.softmax(prediction, dim=1)

        
        pred = output.data.cpu()[0].numpy()  # BxCxHxW
        pred = np.transpose(pred,(1,2,0))

        pred = np.argmax(pred,axis=2)

        cv2.imwrite(r'output/'+img_name[0]+'.png',pred*255)
       

if __name__ == '__main__':
    main()
