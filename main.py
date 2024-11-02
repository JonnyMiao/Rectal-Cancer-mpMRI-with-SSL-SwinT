import argparse
import sys
import time
import numpy as np
import torch
import torch.nn.parallel
from monai.losses import FocalLoss
from visdom import Visdom
from networks.CGMFormer import CGMFormer
from optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from tester import Tester
from trainer import Trainer
from utils.data_loader import get_loader

viz = Visdom()
viz.check_connection()

parser = argparse.ArgumentParser(description='Rectal Image Classification Pipeline')
check_points = None
parser.add_argument('--test_mode', default=False, help='running in test mode or train mode')
parser.add_argument('--task', default='label_TRG', choices=['label_TRG', 'label_pT', 'label_pN'])
parser.add_argument('--pretrained_model_name', default=None, help='whether to use pretrained model')
parser.add_argument('--pre_trained', default=True, help='whether to use pretrained model')
parser.add_argument('--checkpoint', default=check_points, help='start training from saved checkpoint')
parser.add_argument('--dataset', default='RectalMRI', help='dataset name')

# training settings
parser.add_argument('--save_checkpoint', default=True, action='store_true', help='save checkpoint during training')
parser.add_argument('--max_epochs', default=200, type=int, help='max number of training epochs')
parser.add_argument('--batch_size', default=12, type=int, help='number of batch size')
parser.add_argument('--in_channels', default=3, type=int, help='input_channels')
parser.add_argument('--out_channels', default=2, type=int, help='input_channels')
parser.add_argument('--optim_lr', default=1e-5, type=float, help='optimization learning rate')
parser.add_argument('--optim_name', default='adamw', type=str, help='optimization algorithm')
parser.add_argument('--reg_weight', default=1e-6, type=float, help='regularization weight')
parser.add_argument('--momentum', default=0.99, type=float, help='momentum')
parser.add_argument('--noamp', default=False, action='store_true', help='do NOT use amp for training')
parser.add_argument('--val_every', default=5, type=int, help='validation frequency')
parser.add_argument('--workers', default=2, type=int, help='number of workers')

# resolution settings
parser.add_argument('--roi_x', default=128, type=int, help='number of workers')
parser.add_argument('--roi_y', default=128, type=int, help='number of workers')
parser.add_argument('--roi_z', default=32, type=int, help='number of workers')
parser.add_argument('--roi_size', default=(128, 128, 32), type=tuple, help='number of workers')

# model settings
parser.add_argument('--dropout_rate', default=0.2, type=float, help='dropout rate')
parser.add_argument('--lrscheduler', default='cosine_anneal', type=str, help='type of learning rate scheduler')
parser.add_argument('--warmup_ratio', default=0.1, type=float, help='number of warmup epochs')


def datestr():
    now = time.ctime()[4:-5].replace(' ', '_').replace('Sep', '09').replace(':', '_')
    return now


def main():
    args = parser.parse_args()
    args.model = args.model_name
    args.lr = args.optim_lr
    args.warmup_epochs = int(args.max_epochs * args.warmup_ratio)
    args.save = './save_models/' + args.dataset + '_' \
                + args.task + '_' \
                + args.roi + '/' + args.model_name + '_checkpoints/' + args.model_name + '_{}'.format(datestr())
    args.amp = not args.noamp
    args.roi_size = (args.roi_x, args.roi_y, args.roi_z)
    main_worker(gpu=0, args=args)


def main_worker(gpu, args):
    np.set_printoptions(formatter={'float': '{: 0.3f}'.format}, suppress=True)
    args.gpu = gpu
    args.epochs = args.max_epochs
    torch.cuda.set_device(args.gpu)
    torch.backends.cudnn.benchmark = True

    loader = get_loader(args)
    print('Batch size is:', args.batch_size, 'epochs', args.max_epochs)

    model = CGMFormer(
        img_size=args.roi_size,
        patch_size=2,
        embed_dim=64,
        num_classes=args.out_channels
    )
    if args.test_mode:
        checkpoint = torch.load(args.checkpoint)
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in checkpoint['model_state_dict'].items():
            new_state_dict[k.replace('backbone.', '')] = v
        model.load_state_dict(new_state_dict, strict=False)
        model = model.cuda()
        start_epoch = checkpoint['epoch']
        best_auc = checkpoint['Acc']
        print("=> loaded checkpoint '{}' (epoch {}) (bestauc {})".format(args.checkpoint, start_epoch, best_auc))
        print("-----------------Start Testing-----------------")
        tester = Tester(viz=viz, args=args, model=model, test_data_loader=loader)
        tester.running_test()
        sys.exit(0)

    if args.task == 'label_pT':
        loss = torch.nn.CrossEntropyLoss()
    else:
        g = 5 if args.task == 'label_TRG' else 2
        loss = FocalLoss(gamma=g)

    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Total parameters count', pytorch_total_params)

    best_acc = 0
    start_epoch = 0
    if args.optim_name == 'adam':
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=args.optim_lr,
                                     weight_decay=args.reg_weight)
    elif args.optim_name == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(),
                                      lr=args.optim_lr,
                                      weight_decay=args.reg_weight)
    elif args.optim_name == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(),
                                    lr=args.optim_lr,
                                    momentum=args.momentum,
                                    nesterov=True,
                                    weight_decay=args.reg_weight)
    else:
        raise ValueError('Unsupported Optimization Procedure: ' + str(args.optim_name))

    if args.lrscheduler == 'warmup_cosine':
        scheduler = LinearWarmupCosineAnnealingLR(optimizer,
                                                  warmup_epochs=args.warmup_epochs,
                                                  max_epochs=args.max_epochs)
    elif args.lrscheduler == 'cosine_anneal':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                               T_max=args.max_epochs)
    else:
        scheduler = None

    model = model.cuda()
    if args.checkpoint is not None:
        checkpoint = torch.load(args.checkpoint)
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in checkpoint['model_state_dict'].items():
            new_state_dict[k.replace('backbone.', '')] = v
        model.load_state_dict(new_state_dict, strict=False)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
        if 'epoch' in checkpoint:
            start_epoch = checkpoint['epoch']
        if 'Acc' in checkpoint:
            best_acc = checkpoint['Acc']
        print("=> loaded checkpoint '{}' (epoch {}) (BestAcc {})".format(args.checkpoint, start_epoch, best_acc))

    trainer = Trainer(model=model, viz=viz, train_loader=loader[0], val_loader=loader[1], optimizer=optimizer,
                      loss_func=loss, args=args, scheduler=scheduler, start_epoch=start_epoch)
    model, _ = trainer.run_training()
    args.test_mode = True
    loader = get_loader(args)
    print("-----------------Start Testing-----------------")
    tester = Tester(viz=viz, args=args, model=model, test_data_loader=loader)
    tester.running_test()


if __name__ == '__main__':
    main()
