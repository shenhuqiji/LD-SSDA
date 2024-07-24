from datetime import datetime
import os
import os.path as osp

# PyTorch includes
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import argparse
import yaml
from train_process import Trainer
import torch.backends.cudnn as cudnn

# Custom includes
from dataloaders import lanes_dataloader as DL
from dataloaders import custom_transforms as tr
# # from networks.deeplabv3 import *
from networks.erfnet import ERFNet
from networks.GAN import DomainDiscriminator, OutputDiscriminator, DomainDiscriminator_MLP
# from networks.HT import hough_transform

here = osp.dirname(osp.abspath(__file__))

def main():
    parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )
    parser.add_argument('-g', '--gpu', type=int, default=0, help='gpu id')
    parser.add_argument('--resume', default=r'./checkpoint.pth.tar'
                        , help='checkpoint path')

    parser.add_argument(
        '--datasetS', type=str, default='Tusimple', help='source domain images'
    )
    parser.add_argument(
        '--datasetT', type=str, default='VIL100', help='target domain images'
    )
    parser.add_argument(
        '--batch-size', type=int, default=8, help='batch size for training the model'
    )
    parser.add_argument(
        '--group-num', type=int, default=1, help='group number for group normalization'
    )
    parser.add_argument(
        '--max-epoch', type=int, default=60, help='max epoch'
    )
    parser.add_argument(
        '--stop-epoch', type=int, default=60, help='stop epoch'
    )
    parser.add_argument(
        '--interval-validate', type=int, default=1, help='interval epoch number to valide the model'
    )
    parser.add_argument(
        '--lr-gen', type=float, default=1e-3, help='learning rate',
    )
    parser.add_argument(
        '--lr-dis', type=float, default=2.5e-4, help='learning rate',
    )
    parser.add_argument(
        '--lr-decrease-rate', type=float, default=0.1, help='ratio multiplied to initial lr',
    )
    parser.add_argument(
        '--weight-decay', type=float, default=0.0005, help='weight decay',
    )
    parser.add_argument(
        '--momentum', type=float, default=0.99, help='momentum',
    )
    parser.add_argument(
        '--data-dir',
        default=r'./dataset/',
        help='data root path'
    )
    parser.add_argument(
        '--output',
        default=r'./model/',
        help='path of output model',
    )


    args = parser.parse_args()

    # args.model = 'mobilenet'

    now = datetime.now()
    args.out = osp.join(here, 'logs',args.datasetT, now.strftime('%Y%m%d_%H%M%S.%f'))

    os.makedirs(args.out)
    with open(osp.join(args.out, 'config.yaml'), 'w') as f:
        yaml.safe_dump(args.__dict__, f, default_flow_style=False)


    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    cuda = torch.cuda.is_available()

    cudnn.benchmark = True
    cudnn.fastest = True

    # transforms_source = transforms.Compose([tr.RandomFlip(),tr.ToTensor()])

    domain = DL.LanesSegmentation(base_dir=args.data_dir, dataset=args.datasetS, split='train',
                                                         transform=None, domain=None)
    domain_loaderS = DataLoader(domain, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True,drop_last=True)
    domain_T = DL.LanesSegmentation(base_dir=args.data_dir, dataset=args.datasetT, split='labeled_train_50',
                                                             transform=None, domain=None)
    domain_loaderT = DataLoader(domain_T, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True,drop_last=True)
    domain_U = DL.LanesSegmentation(base_dir=args.data_dir, dataset=args.datasetT, split='train',
                                       transform=None, domain='target')
    domain_loaderU = DataLoader(domain_U, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True,drop_last=True)

    ### validation
    domain_U_val = DL.LanesSegmentation(base_dir=args.data_dir, dataset=args.datasetT, split='val',
                                       transform=None, domain=None)
    domain_loaderU_val = DataLoader(domain_U_val, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True,drop_last=True)


    # 2. model
    model_gen = ERFNet(num_classes=2).cuda()

    model_dis = DomainDiscriminator().cuda()
    # model_dis = DomainDiscriminator_MLP().cuda()
    model_dis2 = OutputDiscriminator().cuda()


    start_epoch = 0
    start_iteration = 0

    # 3. optimizer

    optim_gen = torch.optim.Adam(
        model_gen.parameters(),
        lr=args.lr_gen,
        betas=(0.9, 0.99)
    )
    optim_dis = torch.optim.SGD(
        model_dis.parameters(),
        lr=args.lr_dis,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )
    optim_dis2 = torch.optim.SGD(
        model_dis2.parameters(),
        lr=args.lr_dis,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )

    # if args.resume:
    #     checkpoint = torch.load(args.resume)
    #     pretrained_dict = checkpoint['model_state_dict']
    #     model_gen.load_state_dict(pretrained_dict)

    trainer = Trainer.Trainer(
        cuda=cuda,
        model_gen=model_gen,
        model_domain_dis=model_dis,
        model_uncertainty_dis=model_dis2,
        optimizer_gen=optim_gen,
        optimizer_domain_dis=optim_dis,
        optimizer_uncertainty_dis=optim_dis2,
        lr_gen=args.lr_gen,
        lr_dis=args.lr_dis,
        lr_decrease_rate=args.lr_decrease_rate,
        domain_loaderU_val=domain_loaderU_val,
        domain_loaderU=domain_loaderU,
        domain_loaderS=domain_loaderS,
        domain_loaderT=domain_loaderT,
        out=args.output,
        max_epoch=args.max_epoch,
        stop_epoch=args.stop_epoch,
        interval_validate=args.interval_validate,
        batch_size=args.batch_size,
    )
    trainer.epoch = start_epoch
    trainer.iteration = start_iteration
    trainer.train()

if __name__ == '__main__':
    main()
