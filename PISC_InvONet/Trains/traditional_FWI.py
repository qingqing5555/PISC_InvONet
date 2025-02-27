from Functions.data_load import *
from Functions.my_math import *
from Functions.data_plot import *
from Config.Path import *
import torch.optim as optim
import time

"""

edit: suzy 20231128
这个文件是传统反演方法的深度学习实现

"""

# 检查cudnn cuda
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
cuda_available = torch.cuda.is_available()
device = torch.device("cuda" if cuda_available else "cpu")

# ----------------------------加载数据------------------------------
# 1 速度模型
model_true = loadtruemodel(data_path, vmodel_dim).to(device)
# model_true = torch.Tensor(np.load('F:/suzy/OpenFWI/FWI/SEG/model1.npy')[0][0])

# 2 初始模型
init_model_path = ResultPath + str(data_name) + '_initmodel.mat'
if os.path.exists(init_model_path):
    print('存在已构建的初始模型，正在加载...')
    model = load_init_model(init_model_path).clone().to(device)
    model.requires_grad = True
    print(' 真实模型最大波速: ', model_true.max())
    # 把model设置为可训练的参数 这里model是初始模型的clone
    model = torch.nn.Parameter(model)
else:
    raise Exception('不存在初始模型数据，请构建...')

# 3 创建包含源和接收器位置的数组
x_s, x_r = createSR(num_shots, num_sources_per_shot, num_receivers_per_shot, d_source, first_source, d_receiver,
                        first_receiver, source_depth, receiver_depth, device)
x_s, x_r = x_s.to(device), x_r.to(device)

# 4 震源
init_source_filepath = ResultPath + str(data_name) + '_initsource.mat'
if os.path.exists(init_source_filepath):
    print('初始震源已存在,正在加载...')
    source_amplitudes_init, source_amplitudes_true = loadinitsource(init_source_filepath)
    source_amplitudes_init.to(device)
    source_amplitudes_true.to(device)
else:
    raise Exception('不存在初始震源，请构建...')

# 5 地震数据
# 如果有噪声
if AddNoise == True and noise_var != None:
    if noise_type == 'Gaussian':
        ResultPath = ResultPath + 'AWGN_var' + str(noise_var) + '/'
    noise_filepath = ResultPath + str(data_name) + '_noisercv_amps.mat'
    if os.path.exists(noise_filepath):
        print('地震数据（noise）已存在,正在加载...')
        receiver_amplitudes_true = loadrcv(noise_filepath).to(device)
    else:
        raise Exception('不存在地震数据（noise），请构建...')
else:
    # 如果无噪声
    rcv_filepath = ResultPath + str(data_name) + '_rcv_amps.mat'
    if os.path.exists(rcv_filepath):
        print('地震数据（clean）已存在,正在加载...')
        receiver_amplitudes_true = loadrcv(rcv_filepath).to(device)
    else:
        raise Exception('不存在地震数据（clean），请构建...')

# --------------------------超参数-----------------------------------
# 优化器：训练参数，步长，动量参数，数值稳定参数，L2正则化超参数
# my_model = ConvModNet_SEG()
# optimizer = torch.optim.AdamW(model.parameters(), lr=fwi_lr, betas=(0.9, 0.999), weight_decay=1e-4)
optimizer = optim.Adam([{'params': model, 'lr': fwi_lr, 'betas': (0.5, 0.9), 'eps': 1e-8, 'weight_decay': 0}])

if fwi_weight_decay > 0:
    # 优化器对象 多少轮循环更新一次学习率 每次更新lr的gamma倍
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=fwi_stepsize, gamma=fwi_weight_decay)
# 损失函数选择 loss option: L1, L2, 1-D W1
if fwi_loss_type == 'L1':
    criterion = torch.nn.L1Loss()
elif fwi_loss_type == 'L2':
    criterion = torch.nn.MSELoss()
elif fwi_loss_type == 'W1':
    trans_type = 'linear'  # linear, square, exp, softplus, abs
else:
    raise NotImplementedError

# 结果存储路径 损失函数类型-步长-batch-归一化-epoch次数
fwi_result = ResultPath + 'FWI' + '_loss' + str(fwi_loss_type) + '_lr' + str(fwi_lr) + \
             '_batch' + str(fwi_batch) + '_norm' + str(data_norm) + '_epoch' + str(fwi_num_epochs)

if fwi_weight_decay > 0:
    fwi_result = fwi_result + '_dra' + str(fwi_weight_decay) + '_step' + str(fwi_stepsize)

if AddTV:
    fwi_result = fwi_result + '_alp' + str(alpha_tv)

fwi_result = fwi_result + '/'

if not os.path.exists(fwi_result):
    os.makedirs(fwi_result)

rcv_amps_true = receiver_amplitudes_true.clone()

print()
print('*******************************************')
print('          START Traditional FWI            ')
print('*******************************************')
# 初始化参数
SNR = 0.0
SSIM = 0.0
Loss = 0.0
ERROR = 0.0
TOL = 0.0


def fwi_main():
    global model_true, source_amplitudes_init, SNR, SSIM, Loss, ERROR

    t_start = time.time()
    model_true = model_true.view(nz, ny)
    # 每个batch多少shots
    num_shots_per_batch = int(num_shots / fwi_batch)

    for i in range(fwi_num_epochs):
        # loss初始化
        epoch_loss = 0.0

        for it in range(fwi_batch):
            iteration = i * fwi_batch + it + 1
            optimizer.zero_grad()  # 梯度记得清零

            # prop = deepwave.scalar.Propagator({'vp': model}, dx, pml_width, order, survey_pad)
            batch_src_amps = source_amplitudes_init.repeat(1, num_shots_per_batch, 1).to(device)  # 每个batch相应shot的震源数据
            batch_rcv_amps_true = rcv_amps_true[:, it::fwi_batch].to(device)  # 采样的真实地震数据

            batch_x_s = x_s[it::fwi_batch].to(device)
            batch_x_r = x_r[it::fwi_batch].to(device)
            # batch_rcv_amps_pred = prop(batch_src_amps, batch_x_s, batch_x_r, dt)  # 模拟波的传播
            batch_rcv_amps_pred = createdata(model, dx, dt, batch_src_amps, batch_x_s, batch_x_r,
                                             order, pml_width, peak_freq)

            if fwi_loss_type == 'L1' or fwi_loss_type == 'L2':
                if data_norm:
                    # normalize
                    batch_rcv_amps_true = shot_max_normalize(batch_rcv_amps_true.permute(1, 0, 2).unsqueeze(1)).squeeze(
                        1).permute(1, 0, 2) * fscale
                    batch_rcv_amps_pred = shot_max_normalize(batch_rcv_amps_pred.permute(1, 0, 2).unsqueeze(1)).squeeze(
                        1).permute(1, 0, 2) * fscale
                    loss = criterion(batch_rcv_amps_pred, batch_rcv_amps_true)
                else:
                    loss = criterion(batch_rcv_amps_pred, batch_rcv_amps_true)
            elif fwi_loss_type == 'W1':
                loss = Wasserstein1(batch_rcv_amps_pred, batch_rcv_amps_true, trans_type, theta=1.1)
            else:
                raise NotImplementedError

            if fix_value_depth > 0:
                fix_model_grad(fix_value_depth, model)

            epoch_loss += loss.item()
            loss.backward()

            # 裁剪 防止梯度消失 放在backward和step之间
            torch.nn.utils.clip_grad_value_(model, 1e3)

            optimizer.step()

            # 调整张量值区间 使它始终大于0
            model.data = torch.clamp(model.data, min=1e-12)

        if fwi_weight_decay > 0:
            scheduler.step()

        print('Epoch:', i + 1, 'Loss: ', epoch_loss / fwi_batch)
        Loss = np.append(Loss, epoch_loss / fwi_batch)

        # 计算真实和反演之间的差距 SNR,  SSIM ,  RE
        snr = ComputeSNR(model.detach().cpu().numpy(),
                         model_true.detach().cpu().numpy())
        SNR = np.append(SNR, snr)

        ssim = ComputeSSIM(model.detach().cpu().numpy(),
                           model_true.detach().cpu().numpy())
        SSIM = np.append(SSIM, ssim)

        rerror = ComputeRE(model.detach().cpu().numpy(),
                           model_true.detach().cpu().numpy())
        ERROR = np.append(ERROR, rerror)

        if iteration % plot_ite == 0:
            plotcomparison(gt=model_true.cpu().data.numpy(),
                           pre=model.cpu().data.numpy(),
                           ite=iteration, SaveFigPath=fwi_result)
            # plot Loss
            PlotFWILoss(loss=Loss, SaveFigPath=fwi_result)

            # plot SNR, ERROR, and SSIM
            PlotSNR(SNR=SNR, SaveFigPath=fwi_result)
            PlotSSIM(SSIM=SSIM, SaveFigPath=fwi_result)
            PlotERROR(ERROR=ERROR, SaveFigPath=fwi_result)

        if (i + 1) % savepoch == 0 or (i + 1) == fwi_num_epochs:
            # 保存model和loss
            spio.savemat(fwi_result + 'FWIRec_' + str(fwi_loss_type) + '.mat',
                         {'rec': model.cpu().data.numpy()})
            spio.savemat(fwi_result + 'FWIMetric_' + str(fwi_loss_type) + '.mat',
                         {'SNR': SNR, 'SSIM': SSIM,
                          'Loss': Loss, 'ERROR': ERROR})

    t_end = time.time()
    elapsed_time = t_end - t_start
    print('Running complete in {:.0f}m  {:.0f}s'.format(elapsed_time // 60, elapsed_time % 60))
    np.savetxt(fwi_result + 'run_result.txt',
               np.hstack((fwi_num_epochs, elapsed_time // 60, elapsed_time % 60, snr, ssim, rerror)),
               fmt='%5.4f')  # ssim,


if __name__ == "__main__":
    fwi_main()
    exit(0)
