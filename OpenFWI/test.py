import sys
import time
import datetime
import json
import torch.nn as nn
import torchvision
import utils
import network
import transforms as T
import pytorch_ssim
from vis import *
from dataset import FWIDataset
from torchvision.transforms import Compose
from pytorch_msssim import ms_ssim
from torch.utils.data import SequentialSampler
from torch.utils.data.dataloader import default_collate


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


def evaluate(model, criterions, dataloader, device, k, ctx,
             vis_path, vis_batch, vis_sample, missing, std):
    model.eval() 

    label_list, label_pred_list = [], []  # store denormalized predcition & gt in numpy
    label_tensor, label_pred_tensor = [], []  # store normalized prediction & gt in tensor
    if missing or std:
        data_list, data_noise_list = [], []  # store original data and noisy/muted data

    with torch.no_grad():
        batch_idx = 0
        for data, label in dataloader:

            data = data.type(torch.FloatTensor).to(device, non_blocking=True)  # 40 5 1000 700 tensor
            label = label.type(torch.FloatTensor).to(device, non_blocking=True)  # 40 1 70 70

            label_np = T.tonumpy_denormalize(label, ctx['label_min'], ctx['label_max'], exp=False)  # 40 1 70 70 np
            label_list.append(label_np)
            label_tensor.append(label)

            if missing or std:
                # Add gaussian noise
                # data_noise = torch.clip(data + (std ** 0.5) * torch.randn(data.shape).to(device, non_blocking=True),
                #                         min=-1, max=1)
                noise = torch.tensor(np.random.normal(0, std, data.shape), dtype=torch.float32)
                data_noise = torch.clip(data + noise.to(device, non_blocking=True),
                                        min=-22, max=41)

                # Mute some traces
                mute_idx = np.random.choice(data.shape[3], size=missing, replace=False)
                data_noise[:, :, :, mute_idx] = data[0, 0, 0, 0]

                data_np = T.tonumpy_denormalize(data, ctx['data_min'], ctx['data_max'], k=k)
                data_noise_np = T.tonumpy_denormalize(data_noise, ctx['data_min'], ctx['data_max'], k=k)
                data_list.append(data_np)
                data_noise_list.append(data_noise_np)
                pred = model(data_noise)
            else:
                pred = model(data)

            # --------------------------
            # model_FWI = network.model_dict['ConvModNet'](upsample_mode=None,
            #                                            sample_spatial=1.0, sample_temporal=1,
            #                                            norm='bn').to(device)
            # checkpoint = torch.load('F:/suzy/OpenFWI/Invnet_models/curvevel_b/ConvModNet_FWI_step0/checkpoint.pth',
            #                         map_location='cpu')
            # model_FWI.eval()
            # model_FWI = network.load_dict_to_model(checkpoint['model'], model_FWI)
            # data_np_ = T.tonumpy_denormalize(data, ctx['data_min'], ctx['data_max'], k=k)
            # true_data_norm = log_minmaxNormalize(torch.from_numpy(data_np_), ctx['data_min'], ctx['data_max'])
            # pre_model_FWI = model_FWI(true_data_norm.to(device))
            # pred_FWI = denormalize(pre_model_FWI, ctx['label_min'], ctx['label_max'], False)
            # is_same = np.array_equal(pred, pre_model_FWI)   # false?
            # ctxmin = ctx['data_min']
            # ctxmax = ctx['data_max']

            # true_data_norm = log_minmaxNormalize(data, -29.04, 57.03)
            # ----------------------------------------
            label_pred_np = T.tonumpy_denormalize(pred, ctx['label_min'], ctx['label_max'], exp=False)
            label_pred_list.append(label_pred_np)
            label_pred_tensor.append(pred)

            # Visualization
            if vis_path and batch_idx < vis_batch:
                for i in range(vis_sample):
                    plot_velocity_seg(label_pred_np[i, 0], label_np[i, 0],
                                  f'{vis_path}/V_{batch_idx}_{i}.png')  # vmin=ctx['label_min'], vmax=ctx['label_max'])
                    if missing or std:
                        for ch in [2]:  # range(data.shape[1]):
                            plot_seismic(data_np[i, ch], data_noise_np[i, ch], f'{vis_path}/S_{batch_idx}_{i}_{ch}.png',
                                         vmin=ctx['data_min'] * 0.01, vmax=ctx['data_max'] * 0.01)
            batch_idx += 1
    print(batch_idx)
    label, label_pred = np.concatenate(label_list), np.concatenate(label_pred_list)
    label_t, pred_t = torch.cat(label_tensor), torch.cat(label_pred_tensor)
    l1 = nn.L1Loss()
    l2 = nn.MSELoss()
    print(f'MAE: {l1(label_t, pred_t)}')
    print(f'MSE: {l2(label_t, pred_t)}')
    ssim_loss = pytorch_ssim.SSIM(window_size=11)
    print(f'SSIM: {ssim_loss(label_t / 2 + 0.5, pred_t / 2 + 0.5)}')  # (-1, 1) to (0, 1)
    print(
        f'MS_SSIM: {ms_ssim(label_t / 2 + 0.5, pred_t / 2 + 0.5, data_range=1, size_average=True, win_size=5)}')  # (-1, 1) to (0, 1)

    for name, criterion in criterions.items():
        print(f' * Velocity {name}: {criterion(label, label_pred)}')
    #     print(f'   | Velocity 2 layers {name}: {criterion(label[:1000], label_pred[:1000])}')
    #     print(f'   | Velocity 3 layers {name}: {criterion(label[1000:2000], label_pred[1000:2000])}')
    #     print(f'   | Velocity 4 layers {name}: {criterion(label[2000:], label_pred[2000:])}')


def main(args):
    print(args)
    print("torch version: ", torch.__version__)
    print("torchvision version: ", torchvision.__version__)

    utils.mkdir(args.output_path)
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

    print("Loading data")
    print("Loading validation data")
    log_data_min = T.log_transform(ctx['data_min'], k=args.k)
    log_data_max = T.log_transform(ctx['data_max'], k=args.k)
    transform_valid_data = Compose([
        T.LogTransform(k=args.k),
        T.MinMaxNormalize(log_data_min, log_data_max),
    ])

    transform_valid_label = Compose([
        T.MinMaxNormalize(ctx['label_min'], ctx['label_max'])
    ])
    if args.val_anno[-3:] == 'txt':
        dataset_valid = FWIDataset(
            args.val_anno,
            sample_ratio=args.sample_temporal,
            file_size=ctx['file_size'],
            transform_data=transform_valid_data,
            transform_label=transform_valid_label
        )
    else:
        dataset_valid = torch.load(args.val_anno)

    print("Creating data loaders")
    valid_sampler = SequentialSampler(dataset_valid)
    dataloader_valid = torch.utils.data.DataLoader(
        dataset_valid, batch_size=args.batch_size,
        sampler=valid_sampler, num_workers=args.workers,
        pin_memory=True, collate_fn=default_collate)

    print("Creating model")
    if args.model not in network.model_dict:
        print('Unsupported model.')
        sys.exit()

    model = network.model_dict[args.model](upsample_mode=args.up_mode,
                                           sample_spatial=args.sample_spatial, sample_temporal=args.sample_temporal,
                                           norm=args.norm).to(device)

    criterions = {
        'MAE': lambda x, y: np.mean(np.abs(x - y)),
        'MSE': lambda x, y: np.mean((x - y) ** 2)
    }

    if args.resume:
        print(args.resume)
        checkpoint = torch.load(args.resume, map_location='cpu')
        model = network.load_dict_to_model(checkpoint['model'], model) 
        print('Loaded model checkpoint at Epoch {} / Step {}.'.format(checkpoint['epoch'], checkpoint['step']))

    if args.vis:
        # Create folder to store visualization results
        vis_folder = f'visualization_{args.vis_suffix}' if args.vis_suffix else 'visualization'
        vis_path = os.path.join(args.output_path, vis_folder)
        utils.mkdir(vis_path)
    else:
        vis_path = None

    print("Start testing")
    start_time = time.time()
    evaluate(model, criterions, dataloader_valid, device, args.k, ctx,
             vis_path, args.vis_batch, args.vis_sample, args.missing, args.std)
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Testing time {}'.format(total_time_str))


def parse_args():
    import argparse
    # python test.py -ds flatvel-a -n YOUR_DIRECTORY -m InversionNet -v flatvel_a_val.txt -r CHECKPOINT.PTH --vis -vb 2 -vsa 3
    parser = argparse.ArgumentParser(description='FCN Testing')
    parser.add_argument('-d', '--device', default='cuda', help='device')
    parser.add_argument('-ds', '--dataset', default='curvevel-a', type=str, help='dataset name') 
    parser.add_argument('-fs', '--file-size', default=None, type=int, help='number of samples in each npy file')

    # Path related
    parser.add_argument('-ap', '--anno-path', default='split_suzy', help='annotation files location')
    parser.add_argument('-v', '--val-anno', default='curvevel_a_val.txt', help='name of val anno')
    parser.add_argument('-o', '--output-path', default='Invnet_models',
                        help='path to parent folder to save checkpoints')
    parser.add_argument('-n', '--save-name', default='curvevel_a/fcn_normal_step2_no_warm_up',
                        help='folder name for this experiment') 
    parser.add_argument('-s', '--suffix', type=str, default=None, help='subfolder name for this run')

    # Model related
    parser.add_argument('-m', '--model', type=str, default='Inv_ResNet', help='inverse model name')  
    parser.add_argument('-no', '--norm', default='bn',
                        help='normalization layer type, support bn, in, ln (default: bn)')
    parser.add_argument('-um', '--up-mode', default=None,
                        help='upsampling layer mode such as "nearest", "bicubic", etc.')
    parser.add_argument('-ss', '--sample-spatial', type=float, default=1.0, help='spatial sampling ratio')
    parser.add_argument('-st', '--sample-temporal', type=int, default=1, help='temporal sampling ratio')

    # Test related
    parser.add_argument('-b', '--batch-size', default=16, type=int)
    parser.add_argument('-j', '--workers', default=0, type=int, help='number of data loading workers (default: 16)')
    parser.add_argument('--k', default=1, type=float, help='k in log transformation')
    parser.add_argument('-r', '--resume', default='checkpoint.pth', help='resume from checkpoint')
    parser.add_argument('--vis', help='visualization option', action="store_true")
    parser.add_argument('-vsu', '--vis-suffix', default=None, type=str, help='visualization suffix')
    parser.add_argument('-vb', '--vis-batch', help='number of batch to be visualized', default=3, type=int)
    parser.add_argument('-vsa', '--vis-sample', help='number of samples in a batch to be visualized', default=3,
                        type=int)
    parser.add_argument('--missing', default=None, type=int, help='number of missing traces')
    parser.add_argument('--std', default=None, type=float, help='standard deviation of gaussian noise')   # 0.02
    args = parser.parse_args()

    # args.output_path = os.path.join(args.output_path, args.save_name, args.suffix or '')
    # args.val_anno = os.path.join(args.anno_path, args.val_anno)
    # args.resume = os.path.join(args.output_path, args.resume)

    return args


if __name__ == '__main__':
    args = parse_args()

    args.dataset = 'SEG'  
    args.save_name = 'FWI_pretrain'  
    args.model = 'ConvModNet_SEG'  
    args.val_anno = 'SEG_val.txt'  
    args.resume = 'model_40.pth' 
    args.vis_batch = 31  
    args.vis_sample = 1  
    args.vis = True
    args.batch_size = 32

    args.output_path = os.path.join(args.output_path, args.save_name, args.suffix or '')
    args.val_anno = os.path.join(args.anno_path, args.val_anno)
    args.resume = os.path.join(args.output_path, args.resume)

    # *******************
    main(args)
