import wandb
import os
import sys
import time
import datetime
import json

import torch
from torch import nn
from torch.utils.data import RandomSampler, DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.tensorboard import SummaryWriter
import torchvision
from torchvision.transforms import Compose

import utils
import network
from dataset import FWIDataset
from scheduler import WarmupMultiStepLR
import transforms as T
import pytorch_ssim


# 这是用来网格搜索的代码

step = 0


def train_one_epoch(model, criterion, optimizer, lr_scheduler,
                    dataloader, device, epoch, writer, len_training_set, batch_size):
    global step
    model.train()
    loss_total = 0.0
    loss_total_g1v = 0.0
    loss_total_g2v = 0.0

    # Logger setup
    metric_logger = utils.MetricLogger(delimiter='  ')
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value}'))
    metric_logger.add_meter('samples/s', utils.SmoothedValue(window_size=10, fmt='{value:.3f}'))
    header = 'Epoch: [{}]'.format(epoch)

    model.zero_grad()
    for i, (data, label) in enumerate(dataloader):
        start_time = time.time()
        optimizer.zero_grad()
        data, label = data.to(device), label.to(device)
        output = model(data)
        loss, loss_g1v, loss_g2v = criterion(output, label)
        loss.backward()

        loss_total = loss_total * i / (i + 1) + loss / (i + 1)
        loss_total_g1v = loss_total_g1v * i / (i + 1) + loss_g1v / (i + 1)
        loss_total_g2v = loss_total_g2v * i / (i + 1) + loss_g2v / (i + 1)

        if (i + 1) % int(batch_size / 16) == 0 or (i + 1) == len_training_set:
            optimizer.step()
            model.zero_grad()
            lr_scheduler.step()
            step += 1
            if writer:
                writer.add_scalar('lr', lr_scheduler.get_last_lr()[0], step)

            metric_logger.update(loss=loss_total.item(), loss_g1v=loss_total_g1v.item(),
                                 loss_g2v=loss_total_g2v.item(), lr=optimizer.param_groups[0]['lr'])
            metric_logger.meters['samples/s'].update(16 / (time.time() - start_time))
            if writer:
                writer.add_scalar('loss', loss_total.item(), step)
                # writer.add_scalar('loss_g1v', loss_total_g1v.item(), step)
                writer.add_scalar('loss_g2v', loss_total_g2v.item(), step)
    return model, lr_scheduler.get_last_lr()[0], loss_total

def evaluate(model, dataloader, device, writer):
    model.eval()  # 将模型设置为评估模式
    label_tensor, label_pred_tensor = [], []  # 存真实和预测
    with torch.no_grad():
        for data, label in dataloader:
            data = data.type(torch.FloatTensor).to(device, non_blocking=True)  # 40 5 1000 700 tensor
            label = label.type(torch.FloatTensor).to(device, non_blocking=True)  # 40 1 70 70

            label_tensor.append(label)
            pred = model(data)

            label_pred_tensor.append(pred)

    label_t, pred_t = torch.cat(label_tensor), torch.cat(label_pred_tensor)
    l1 = nn.L1Loss()
    l2 = nn.MSELoss()
    ssim_loss = pytorch_ssim.SSIM(window_size=11)

    my_MAE = l1(label_t, pred_t)
    my_MSE = l2(label_t, pred_t)
    my_SSIM = ssim_loss(label_t / 2 + 0.5, pred_t / 2 + 0.5)

    print(f'MAE: {my_MAE}')
    print(f'MSE: {my_MSE}')
    print(f'SSIM: {my_SSIM}')  # (-1, 1) to (0, 1)

    if writer:
        writer.add_scalar('loss_l1_mae', my_MAE.item(), step)
        writer.add_scalar('loss_l2_mse', my_MSE.item(), step)
        writer.add_scalar('loss_ssim', my_SSIM.item(), step)
    return my_MAE, my_MSE, my_SSIM

# ---------------------------------函数定义-----------------------------------
def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='FCN Training')
    parser.add_argument('-d', '--device', default='cuda', help='device')
    parser.add_argument('-ds', '--dataset', default='curvevel-a', type=str, help='dataset name')
    parser.add_argument('-fs', '--file-size', default=500, type=int, help='number of samples in each npy file')

    # Path related
    parser.add_argument('-ap', '--anno-path', default='split_suzy', help='annotation files location')
    parser.add_argument('-t', '--train-anno', default='curvevel_a_train.txt', help='name of train anno')
    parser.add_argument('-v', '--val-anno', default='curvevel_a_val.txt', help='name of val anno')
    parser.add_argument('-o', '--output-path', default='Invnet_models/',
                        help='path to parent folder to save checkpoints')
    parser.add_argument('-l', '--log-path', default='Invnet_models', help='path to parent folder to save logs')
    parser.add_argument('-n', '--save-name', default='curvevel_a/',
                        help='folder name for this experiment')
    parser.add_argument('-s', '--suffix', type=str, default=None, help='subfolder name for this run')

    # Model related
    # parser.add_argument('-m', '--model', type=str, default='RBF_Net', help='inverse model name')  # 网络
    parser.add_argument('-um', '--up-mode', default=None,
                        help='upsampling layer mode such as "nearest", "bicubic", etc.')
    parser.add_argument('-ss', '--sample-spatial', type=float, default=1.0, help='spatial sampling ratio')
    parser.add_argument('-st', '--sample-temporal', type=int, default=1, help='temporal sampling ratio')
    # Training related
    parser.add_argument('-b', '--batch-size', default=256, type=int)
    # parser.add_argument('--lr', default=0.0004, type=float, help='initial learning rate')
    parser.add_argument('-lm', '--lr-milestones', nargs='+', default=[], type=int, help='decrease lr on milestones')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    # parser.add_argument('--weight-decay', default=1e-4, type=float, help='weight decay (default: 1e-4)')
    # parser.add_argument('--lr-gamma', default=0.1, type=float, help='decrease lr by a factor of lr-gamma')
    parser.add_argument('--lr-warmup-epochs', default=0, type=int, help='number of warmup epochs')
    parser.add_argument('-eb', '--epoch_block', type=int, default=20, help='epochs in a saved block')
    parser.add_argument('-nb', '--num_block', type=int, default=3, help='number of saved block')
    parser.add_argument('-j', '--workers', default=0, type=int, help='number of data loading workers (default: 16)')
    parser.add_argument('--k', default=1, type=float, help='k in log transformation')
    parser.add_argument('--print-freq', default=50, type=int, help='print frequency')
    parser.add_argument('-r', '--resume', default="checkpoint.pth", help='resume from checkpoint')  # "checkpoint.pth"
    parser.add_argument('--start-epoch', default=0, type=int, help='start epoch')

    # Loss related
    parser.add_argument('-g1v', '--lambda_g1v', type=float, default=1.0)
    parser.add_argument('-g2v', '--lambda_g2v', type=float, default=0)

    # Distributed training related
    parser.add_argument('--sync-bn', action='store_true', help='Use sync batch norm')
    parser.add_argument('--world-size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')

    # Tensorboard related
    parser.add_argument('--tensorboard', action='store_true', help='Use tensorboard for logging.')

    args = parser.parse_args()

    # args.output_path = os.path.join(args.output_path, args.save_name, args.suffix or '')
    # args.log_path = os.path.join(args.log_path, args.save_name, args.suffix or '')
    args.train_anno = os.path.join(args.anno_path, args.train_anno)
    args.val_anno = os.path.join(args.anno_path, args.val_anno)
    args.epochs = args.epoch_block * args.num_block

    if args.resume:
        # args.resume = os.path.join(args.output_path, args.resume)
        args.resume = 'F:/suzy/OpenFWI/Invnet_models/curvevel_a/InversionNet_step0_5000/model_860.pth'


    return args

def my_train():
    global step
    args = parse_args()

    with wandb.init():
        config = wandb.config
        step = 0
        seed = config.retrain
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        print(args)
        print('torch version: ', torch.__version__)
        print('torchvision version: ', torchvision.__version__)

        args.output_path = os.path.join(args.output_path, args.save_name, 'model_' + str(config.model) + 'lr_' + str(config.lr) + 'gamma_' + str(config.gamma))
        utils.mkdir(args.output_path)  # 存储的文件夹

        # Set up tensorboard summary writer
        train_writer, val_writer = None, None
        args.tensorboard = True
        if args.tensorboard:
            train_writer = SummaryWriter(os.path.join(args.output_path, 'logs', 'train'))
            val_writer = SummaryWriter(os.path.join(args.output_path, 'logs', 'val'))


        device = torch.device(args.device)
        torch.backends.cudnn.benchmark = True

        with open('dataset_config.json') as f:
            try:
                ctx = json.load(f)[args.dataset]
            except KeyError:
                print('Unsupported dataset.')
                sys.exit()

        if args.file_size is not None:
            ctx['file_size'] = args.file_size

        # Create dataset and dataloader
        print('Loading data')
        print('Loading training data')

        # Normalize data and label to [-1, 1]
        data_min = ctx['data_min']
        data_max = ctx['data_max']
        transform_data = Compose([
            T.LogTransform(k=args.k),
            T.MinMaxNormalize(T.log_transform(data_min, k=args.k), T.log_transform(data_max, k=args.k))
        ])
        label_min = ctx['label_min']
        label_max = ctx['label_max']
        transform_label = Compose([
            T.MinMaxNormalize(label_min, label_max)
        ])
        if args.train_anno[-3:] == 'txt':
            dataset_train = FWIDataset(
                args.train_anno,
                preload=True,
                sample_ratio=args.sample_temporal,
                file_size=ctx['file_size'],
                transform_data=transform_data,
                transform_label=transform_label
            )
        else:
            dataset_train = torch.load(args.train_anno)

        print('Loading validation data')
        if args.val_anno[-3:] == 'txt':
            dataset_valid = FWIDataset(
                args.val_anno,
                preload=True,
                sample_ratio=args.sample_temporal,
                file_size=ctx['file_size'],
                transform_data=transform_data,
                transform_label=transform_label
            )
        else:
            dataset_valid = torch.load(args.val_anno)

        print('Creating data loaders')
        train_sampler = RandomSampler(dataset_train)
        valid_sampler = RandomSampler(dataset_valid)

        dataloader_train = DataLoader(
            dataset_train, batch_size=16,
            sampler=train_sampler, num_workers=args.workers,
            pin_memory=True, drop_last=True, collate_fn=default_collate)

        dataloader_valid = DataLoader(
            dataset_valid, batch_size=16,
            sampler=valid_sampler, num_workers=args.workers,
            pin_memory=True, collate_fn=default_collate)

        print('Creating model')
        if config.model not in network.model_dict:
            print('Unsupported model.')
            sys.exit()
        model = network.model_dict[config.model](upsample_mode=args.up_mode,
                                               sample_spatial=args.sample_spatial,
                                               sample_temporal=args.sample_temporal).to(
            device)

        # Define loss function
        l1loss = nn.L1Loss()
        l2loss = nn.MSELoss()

        def criterion(pred, gt):
            loss_g1v = l1loss(pred, gt)
            loss_g2v = l2loss(pred, gt)
            loss = args.lambda_g1v * loss_g1v + args.lambda_g2v * loss_g2v
            return loss, loss_g1v, loss_g2v

        # Scale lr according to effective batch size
        lr = config.lr
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=config.weight_decay)

        # Convert scheduler to be per iteration instead of per epoch
        scheduler_string = "step"
        if scheduler_string == "cyclic":
            lr_scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=lr, max_lr=0.001, cycle_momentum=False,
                                                             step_size_up=int(16 / 2) * args.epochs,
                                                             mode="triangular2")
        elif scheduler_string == "step":
            lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=16 * 20, gamma=config.gamma)
        elif scheduler_string == "warmup":
            warmup_iters = args.lr_warmup_epochs * len(dataloader_train)
            lr_milestones = [len(dataloader_train) * m for m in args.lr_milestones]
            lr_scheduler = WarmupMultiStepLR(
                optimizer, milestones=lr_milestones, gamma=config.gamma,
                warmup_iters=warmup_iters, warmup_factor=1e-5)

        model_without_ddp = model

        best_loss = 100
        args.start_epoch = 0
        if args.resume:
            print("loading checkpoint")
            checkpoint = torch.load(args.resume, map_location='cpu')
            model_without_ddp = network.load_dict_to_model(checkpoint['model'], model_without_ddp)
            # optimizer.load_state_dict(checkpoint['optimizer'])
            # lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1
            step = checkpoint['step']
            # lr_scheduler.milestones = lr_milestones
            # best_loss = checkpoint['best_loss']

        print('Start training')
        start_time = time.time()
        chp = 1
        patience = 6
        patience_counter = 0
        print('Start training epoch:', args.start_epoch)
        print('Total training epoch:', args.start_epoch + args.epochs)
        for epoch in range(args.start_epoch, args.start_epoch + args.epochs):
            model_without_ddp, t_lr, t_loss = train_one_epoch(model_without_ddp, criterion, optimizer, lr_scheduler, dataloader_train,
                                                device, epoch, train_writer, len(dataset_train), args.batch_size)
            loss, my_MSE, my_SSIM = evaluate(model_without_ddp, dataloader_valid, device, val_writer)

            wandb.log({"train_loss": t_loss, "epoch": epoch})
            wandb.log({"Lr": t_lr, "epoch": epoch})
            wandb.log({"val_loss": loss, "epoch": epoch})
            wandb.log({"val_ssim": my_SSIM, "epoch": epoch})

            checkpoint = {
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch,
                'step': step,
                'args': args,
                'config': config,
                'best_loss': loss
            }
            # Save checkpoint per epoch
            if loss < best_loss:
                patience_counter = 0
                # torch.save(model_without_ddp, os.path.join(args.output_path, 'checkpoint.pth'))
                # print('saving checkpoint at epoch: ', epoch)
                chp = epoch
                best_loss = loss
            else:
                patience_counter = patience_counter + 1
                print("patience:", patience_counter)
            # Save checkpoint every epoch block
            print('current best loss: ', best_loss)
            print('current best epoch: ', chp)
            # if args.output_path and (epoch + 1) % args.epoch_block == 0:
            #     utils.save_on_master(
            #         checkpoint,
            #         os.path.join(args.output_path, 'model_{}.pth'.format(epoch + 1)))
            # if patience_counter > patience:
            #     print("Early stopping:", epoch, patience_counter)
            #     break

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Training time {}'.format(total_time_str))


def main():
    # ------------------一些常规设置----------------------------
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available():
        memory_avail = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3  # 4.9998
        print("Running on ", torch.cuda.get_device_name(0), "Total memory: ", memory_avail, " GB")

    # -------------------------随机参数列表---------------------------------
    sweep_config = {
        'method': 'grid'
    }
    metric = {
        'name': 'val_loss',
        'goal': 'minimize'
    }
    sweep_config['metric'] = metric

    training_properties_ = {
        'model': {
            'values': ["InversionNet"]
        },
        'gamma': {
            'values': [1]
        },
        "lr": {
            'values': [0.01, 0.005, 0.001, 0.0005, 0.0001, 0.00005]
        },
        # "lr": {
        #     'distribution': 'uniform',
        #     'min': 0.005,
        #     'max': 0.1
        # },
        "weight_decay": {
            'values': [1e-4]
        },
        "retrain": {
            'values': [888]
        },
        "scheduler": {
            'values': ["step"]
        }
    }
    sweep_config['parameters'] = training_properties_

    print(sweep_config)
    # ----------------------------------
    sweep_id = wandb.sweep(sweep_config, project="pytorch-baseline")
    wandb.agent(sweep_id, my_train, count=6)

if __name__ == '__main__':
    main()

