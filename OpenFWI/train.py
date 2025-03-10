import os
import sys
import time
import datetime
import json
import torch
from torch import nn
from torch.utils.data import RandomSampler, DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel
from torch.utils.tensorboard import SummaryWriter
import torchvision
from torchvision.transforms import Compose
import utils
import network
from dataset import FWIDataset
from scheduler import WarmupMultiStepLR
import transforms as T
import pytorch_ssim


step = 0

#------------------------normalize----------------------#
def log_transform(data, k=1, c=0):
    # log
    log = (torch.log1p(torch.abs(k * data) + c)) * torch.sign(data)
    return log


def MinMaxNormalize(data, min, max, scale=2):
    data = data - min
    data = data / (max - min)  # 0-1
    return (data - 0.5) * 2 if scale == 2 else data  # -1 - 1


def log_minmaxNormalize(data, min, max, scale=2):
    return MinMaxNormalize(log_transform(data), log_transform(torch.tensor(min)), log_transform(torch.tensor(max)))

def exp_transform(data, k=1, c=0):
    return (torch.expm1(torch.abs(data)) - c) * torch.sign(data) / k


def minmax_denormalize(data, min, max, scale=2):
    if scale == 2:
        data = data / 2 + 0.5
    return data * (max - min) + min


def denormalize(data, min, max, exp=True, k=1, c=0, scale=2):
    if exp:
        min = log_transform(torch.tensor(min), k=k, c=c)
        max = log_transform(torch.tensor(max), k=k, c=c)
    data = minmax_denormalize(data, min, max, scale)
    return exp_transform(data, k=k, c=c) if exp else data

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

    # # --------------------------
    # # Forward modeling parameters
    # dx = 10.0
    # n_shots = 5
    # n_sources_per_shot = 1
    # d_source = 17.25  # 17.25 * 10m = 172.5m
    # first_source = 0  # 0 * 10m = 0m
    # source_depth = 0  # 0m
    #
    # n_receivers_per_shot = 35
    # d_receiver = 2  # 1 * 10m = 10m
    # first_receiver = 0  # 0 * 4m = 0m
    # receiver_depth = 0  # 0m
    #
    # freq = 15
    # nt = 1000
    # dt = 0.001
    # peak_time = 1.5 / freq
    #
    # data_min = -29.04  # -26 float
    # data_max = 57.03  # 51
    # model_min = 1500
    # model_max = 4500
    # source_locations = torch.zeros(n_shots, n_sources_per_shot, 2,
    #                                dtype=torch.long, device=device)
    # source_locations[..., 0] = source_depth
    # source_locations[:, 0, 1] = (torch.arange(n_shots) * d_source +
    #                              first_source)
    # # Receiver
    # receiver_locations = torch.zeros(n_shots, n_receivers_per_shot, 2,
    #                                  dtype=torch.long, device=device)
    # receiver_locations[..., 0] = receiver_depth
    # receiver_locations[:, :, 1] = (
    #     (torch.arange(n_receivers_per_shot) * d_receiver +
    #      first_receiver)
    #     .repeat(n_shots, 1)
    # )
    # source_amplitudes = (
    #     (deepwave.wavelets.ricker(freq, nt, dt, peak_time))
    #     .repeat(n_shots, n_sources_per_shot, 1).to(device)
    # )
    # # --------------------------------------------------------

    model.zero_grad()
    for i, (data, label) in enumerate(dataloader):
        start_time = time.time()
        optimizer.zero_grad()
        data, label = data.to(device), label.to(device)
        output = model(data)  # 16 1 70 70

        # # ------------------------PHY--------------------
        # data_pre_list = []
        # arr_sample = np.arange(16)
        # idx = np.random.choice(arr_sample, 3)
        # output_sample = output.clone()[idx, :, :, :]  # Sampling completed
        # output_sample_denorm = denormalize(output_sample, model_min, model_max, False)  # 3 1 70 70
        #
        # for sample_id in range(3):
        #     observed_data = scalar(output_sample_denorm[sample_id][0], dx, dt,
        #                            source_amplitudes=source_amplitudes.to(device),
        #                            source_locations=source_locations.to(device),
        #                            receiver_locations=receiver_locations.to(device),
        #                            accuracy=4,
        #                            pml_freq=freq)
        #     data_pre_list.append(torch.unsqueeze((observed_data[-1] * (-1)).permute(0, 2, 1), 0))   # 5 1000 35
        # data_pre = torch.cat(data_pre_list, dim=0)  # 3 3 1000 35
        # data_pre_norm = log_minmaxNormalize(data_pre, data_min, data_max)
        # -------------------------------------------------
        loss, loss_g1v, loss_g2v, loss_msssim_l1 = criterion(output, label)
        # -----------PHY loss-----------------------
        # data_sample = data[:, :, :, np.arange(0, 70, 2)]
        # loss_Phy, loss_g1v_Phy, loss_g2v_Phy, loss_msssim_l1_Phy = criterion(data_pre_norm, data_sample[idx, :, :, :])
        # loss = 0.5 * loss + 0.5 * loss_Phy
        # ------------------------------------
        loss.backward()

        loss_total = loss_total * i / (i + 1) + loss.item() / (i + 1)
        loss_total_g1v = loss_total_g1v * i / (i + 1) + loss_g1v.item() / (i + 1)
        loss_total_g2v = loss_total_g2v * i / (i + 1) + loss_g2v.item() / (i + 1)

        if (i + 1) % int(batch_size / 16) == 0 or (i + 1) == len_training_set:
            optimizer.step()
            model.zero_grad()
            lr_scheduler.step()
            step += 1
            if writer:
                writer.add_scalar('lr', lr_scheduler.get_last_lr()[0], step)

            metric_logger.update(loss=loss_total, loss_g1v=loss_total_g1v,
                                 loss_g2v=loss_total_g2v, lr=optimizer.param_groups[0]['lr'])
            metric_logger.meters['samples/s'].update(16 / (time.time() - start_time))
            if writer:
                writer.add_scalar('loss', loss_total, step)
                # writer.add_scalar('loss_g1v', loss_total_g1v.item(), step)
                writer.add_scalar('loss_g2v', loss_total_g2v, step)
    return model

def evaluate(model, dataloader, device, writer):
    model.eval() 
    label_tensor, label_pred_tensor = [], [] 
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


def main(args):
    global step
    # random.seed(args.seed)
    # np.random.seed(args.seed)
    torch.manual_seed(8)
    torch.cuda.manual_seed(8)

    print(args)
    print('torch version: ', torch.__version__)
    print('torchvision version: ', torchvision.__version__)

    utils.mkdir(args.output_path)  # create folder to store checkpoints
    utils.init_distributed_mode(args)  # distributed mode initialization

    # Set up tensorboard summary writer
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
    # data_min = -144.94
    # data_max = 279.48
    transform_data = Compose([
        T.LogTransform(k=args.k),
        T.MinMaxNormalize(T.log_transform(data_min, k=args.k), T.log_transform(data_max, k=args.k))
    ])
    label_min = ctx['label_min']
    label_max = ctx['label_max']
    # label_min = 1471
    # label_max = 5772
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
    if args.distributed:
        train_sampler = DistributedSampler(dataset_train, shuffle=True)
        valid_sampler = DistributedSampler(dataset_valid, shuffle=True)
    else:
        train_sampler = RandomSampler(dataset_train)
        valid_sampler = RandomSampler(dataset_valid)

    dataloader_train = DataLoader(
        dataset_train, batch_size=16, shuffle=True,
        num_workers=args.workers,
        pin_memory=True, drop_last=True, collate_fn=default_collate)

    dataloader_valid = DataLoader(
        dataset_valid, batch_size=16, shuffle=True,
        num_workers=args.workers,
        pin_memory=True, collate_fn=default_collate)

    print('Creating model')
    if args.model not in network.model_dict:
        print('Unsupported model.')
        sys.exit()
    model = network.model_dict[args.model](upsample_mode=args.up_mode,
                                           sample_spatial=args.sample_spatial, sample_temporal=args.sample_temporal).to(
        device)

    if args.distributed and args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    # Define loss function
    l1loss = nn.L1Loss()
    l2loss = nn.MSELoss()
    ssim_l1 = pytorch_ssim.MS_SSIM_L1_LOSS()

    def criterion(pred, gt):
        loss_g1v = l1loss(pred, gt)
        loss_g2v = l2loss(pred, gt)
        # loss_ssim_l1 = 1 - ms_ssim(gt / 2 + 0.5, pred / 2 + 0.5, data_range=1, size_average=True, win_size=5)
        loss_ssim_l1 = 0

        loss = args.lambda_g1v * loss_g1v + args.lambda_g2v * loss_ssim_l1
        return loss, loss_g1v, loss_g2v, loss_ssim_l1

    # Scale lr according to effective batch size
    lr = args.lr * args.world_size
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=args.weight_decay)

    # Convert scheduler to be per iteration instead of per epoch
    scheduler_string = "warmup"
    if scheduler_string == "cyclic":
        lr_scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=lr, max_lr=0.001, cycle_momentum=False,
                                                      step_size_up=int(16 / 2) * args.epochs,
                                                      mode="triangular2")
    elif scheduler_string == "step":
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=16*30, gamma=0.96)
    elif scheduler_string == "warmup":
        warmup_iters = args.lr_warmup_epochs * len(dataloader_train)
        lr_milestones = [len(dataloader_train) * m for m in args.lr_milestones]
        lr_scheduler = WarmupMultiStepLR(
            optimizer, milestones=lr_milestones, gamma=args.lr_gamma,
            warmup_iters=warmup_iters, warmup_factor=1e-5)

    model_without_ddp = model
    if args.distributed:
        model = DistributedDataParallel(model, device_ids=[args.local_rank])
        model_without_ddp = model.module

    best_loss = 1.43
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
    count = 0
    print('Start training epoch:', args.start_epoch)
    print('Total training epoch:', args.start_epoch+args.epochs)
    for epoch in range(args.start_epoch, args.start_epoch + args.epochs):
        # if args.distributed:
        #     train_sampler.set_epoch(epoch)
        model_without_ddp = train_one_epoch(model_without_ddp, criterion, optimizer, lr_scheduler, dataloader_train,
                        device, epoch, train_writer, len(dataset_train), args.batch_size)
        loss, my_MSE, my_SSIM = evaluate(model_without_ddp, dataloader_valid, device, val_writer)

        checkpoint = {
            'model': model_without_ddp.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'epoch': epoch,
            'step': step,
            'args': args,
            'best_loss': loss
        }
        # Save checkpoint per epoch
        utils.save_on_master(
            checkpoint,
            os.path.join(args.output_path, 'checkpoint.pth'))
        print('saving checkpoint at epoch: ', epoch)
        if loss < best_loss:
            patirnce = 0
            chp = epoch
            best_loss = loss
        else:
            count = count + 1
            print('count:', count)
        # Save checkpoint every epoch block
        print('current best loss: ', best_loss)
        print('current best epoch: ', chp)
        if args.output_path and (epoch + 1) % args.epoch_block == 0:
            utils.save_on_master(
                checkpoint,
                os.path.join(args.output_path, 'model_{}.pth'.format(epoch + 1)))
        # if count > patience:
        #     print("Early stopping:", epoch, count)
        #     break

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='FCN Training')
    parser.add_argument('-d', '--device', default='cuda', help='device')
    parser.add_argument('-ds', '--dataset', default='curvevel-a', type=str, help='dataset name')  # curvevel-a
    parser.add_argument('-fs', '--file-size', default=500, type=int, help='number of samples in each npy file')

    # Path related
    parser.add_argument('-ap', '--anno-path', default='split_suzy', help='annotation files location')
    parser.add_argument('-t', '--train-anno', default='curvevel_a_train.txt', help='name of train anno')  # 修改训练数据路径
    parser.add_argument('-v', '--val-anno', default='curvevel_a_val.txt', help='name of val anno')
    parser.add_argument('-o', '--output-path', default='Invnet_models',
                        help='path to parent folder to save checkpoints')
    parser.add_argument('-l', '--log-path', default='Invnet_models', help='path to parent folder to save logs')
    parser.add_argument('-n', '--save-name', default='FWI_pretrain_2',
                        help='folder name for this experiment')
    parser.add_argument('-s', '--suffix', type=str, default=None, help='subfolder name for this run')

    # Model related
    parser.add_argument('-m', '--model', type=str, default='InversionNet', help='inverse model name')  # 网络 InversionNet ConvModNet_SEG Df_ConvModNet NO_Df_ConvModNet
    parser.add_argument('-um', '--up-mode', default=None,
                        help='upsampling layer mode such as "nearest", "bicubic", etc.')
    parser.add_argument('-ss', '--sample-spatial', type=float, default=1.0, help='spatial sampling ratio')
    parser.add_argument('-st', '--sample-temporal', type=int, default=1, help='temporal sampling ratio')
    # Training related
    parser.add_argument('-b', '--batch-size', default=256, type=int)
    parser.add_argument('--lr', default=0.0001, type=float, help='initial learning rate')
    parser.add_argument('-lm', '--lr-milestones', nargs='+', default=[], type=int, help='decrease lr on milestones')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--weight-decay', default=1e-4, type=float, help='weight decay (default: 1e-4)')
    parser.add_argument('--lr-gamma', default=0.1, type=float, help='decrease lr by a factor of lr-gamma')
    parser.add_argument('--lr-warmup-epochs', default=0, type=int, help='number of warmup epochs')
    parser.add_argument('-eb', '--epoch_block', type=int, default=10, help='epochs in a saved block')
    parser.add_argument('-nb', '--num_block', type=int, default=2, help='number of saved block')
    parser.add_argument('-j', '--workers', default=0, type=int, help='number of data loading workers (default: 16)')
    parser.add_argument('--k', default=1, type=float, help='k in log transformation')
    parser.add_argument('--print-freq', default=50, type=int, help='print frequency')
    parser.add_argument('-r', '--resume', default=None, help='resume from checkpoint')  # "checkpoint.pth"
    parser.add_argument('--start-epoch', default=0, type=int, help='start epoch')


    # Loss related
    parser.add_argument('-g1v', '--lambda_g1v', type=float, default=1.0)
    parser.add_argument('-g2v', '--lambda_g2v', type=float, default=0)
    parser.add_argument('-phy', '--phy', default=False, help='subfolder name for this run')


    # Distributed training related
    parser.add_argument('--sync-bn', action='store_true', help='Use sync batch norm')
    parser.add_argument('--world-size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')

    # Tensorboard related
    parser.add_argument('--tensorboard', action='store_true', help='Use tensorboard for logging.')

    args = parser.parse_args()

    args.output_path = os.path.join(args.output_path, args.save_name, args.suffix or '')
    args.log_path = os.path.join(args.log_path, args.save_name, args.suffix or '')
    args.train_anno = os.path.join(args.anno_path, args.train_anno)
    args.val_anno = os.path.join(args.anno_path, args.val_anno)
    args.epochs = args.epoch_block * args.num_block

    if args.resume:
        args.resume = os.path.join(args.output_path, args.resume)

    return args


if __name__ == '__main__':
    args = parse_args()
    main(args)

# def my_train():
#     args = parse_args()
#     main(args)
