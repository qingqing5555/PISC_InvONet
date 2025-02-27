from Functions.data_load import *
from Functions.data_plot import *
from Config.Path import *

"""

edit: suzy 20231202

这个文件可以根Path中设置的【速度模型路径、初始模型类型、噪声参数、正演参数】来生成对应的地震资料

"""

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
# --------------------------生成数据---------------------------------------------------------#
# 检查cudnn cuda
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
cuda_available = torch.cuda.is_available()
device = torch.device("cuda" if cuda_available else "cpu")
# 加载真实模型数据
model_true = loadtruemodel(data_path, vmodel_dim)
# model_true = torch.Tensor(np.load('../../OpenFWI/FWI/SEG/model1.npy')[0][0])
# 加载/构建初始模型（真实模型+gfsigma） to(device)
if not os.path.exists(ResultPath):
    os.makedirs(ResultPath)
init_model_path = ResultPath + str(data_name) + '_initmodel.mat'
if os.path.exists(init_model_path):
    print('初始模型已存在...')
else:
    print('准备开始构建初始模型...')
    model_init = createInitialModel(model_true, gfsigma, lipar, fix_value_depth)
    # model = model_init.clone()
    vmin, vmax = np.percentile(model_true.cpu().data.numpy(), [2, 98])  # 计算一个多维数组的任意百分比分位数
    plotinitmodel(model_init, model_true, vmin, vmax, ResultPath)
    spio.savemat(init_model_path,
                 {'initmodel': model_init.cpu().data.numpy()})
    print('初始模型保存成功...')

# 构建震源、地震数据
# 震源
init_source_filepath = ResultPath + str(data_name) + '_initsource.mat'
if os.path.exists(init_source_filepath):
    print('初始震源已存在...')
    source_amplitudes_init, source_amplitudes_true = loadinitsource(init_source_filepath)
else:
    # 生成震源 [nt=1000, num_shots=30, num_sources_per_shot=1] ###
    print('正在生成震源数据...')
    source_amplitudes_true = createSourceAmp(peak_freq, nt, dt, peak_source_time, num_shots)
    source_amplitudes_true = torch.from_numpy(source_amplitudes_true)
    # 如果使用滤波器
    if use_filter:
        source_amplitudes_filt = createFilterSourceAmp(peak_freq, nt, dt, peak_source_time,
                                                       num_shots,
                                                       use_filter, filter_type, freqmin, freqmax,
                                                       corners, df)

        source_amplitudes_filt = torch.from_numpy(source_amplitudes_filt)
        # 获取初始滤波震源
        source_amplitudes_init = source_amplitudes_filt[:, 0, 0].reshape(-1, 1, 1)
    else:
        # 获取初始震源
        source_amplitudes_init = source_amplitudes_true[:, 0, 0].reshape(-1, 1, 1)
    spio.savemat(init_source_filepath,
                 {'initsource': source_amplitudes_init.cpu().data.numpy(),
                  'truesource': source_amplitudes_true.cpu().data.numpy()})

    plotinitsource(init=source_amplitudes_init.cpu().detach().numpy(),
                   gt=source_amplitudes_true[:, 0, 0].cpu().detach().numpy(), SaveFigPath=ResultPath)
    plotsourcespectra(init_source=source_amplitudes_init.cpu().detach().numpy(),
                      true_source=source_amplitudes_true[:, 0, 0].cpu().detach().numpy(),
                      SaveFigPath=ResultPath)
# 地震数据
rcv_filepath = ResultPath + str(data_name) + '_rcv_amps.mat'
if os.path.exists(rcv_filepath):
    print('地震数据已存在...')
    receiver_amplitudes_true = loadrcv(rcv_filepath)
    plotoneshot(receiver_amplitudes_true, ResultPath)
else:
    print('正在生成地震数据.........')
    # 生成地震数据
    #  over x_s=10... x_r=0...399   mar_big:x_s=110...560  x_r=0...566
    x_s, x_r = createSR(num_shots, num_sources_per_shot, num_receivers_per_shot, d_source, first_source, d_receiver,
                        first_receiver, source_depth, receiver_depth, device)
    model_true = model_true.to(device)
    source_amplitudes_true = source_amplitudes_true.to(device)
    receiver_amplitudes = createdata(model_true, dx, dt, source_amplitudes_true, x_s, x_r,
               order, pml_width, peak_freq)
    receiver_amplitudes_true = receiver_amplitudes
    plotoneshot(receiver_amplitudes_true, ResultPath)
    # save the receiver amplitudes
    spio.savemat(rcv_filepath,
                 {'true': receiver_amplitudes_true.cpu().data.numpy(),
                  'rcv': receiver_amplitudes.cpu().data.numpy()})

# [nt, num_shots, num_receivers_per_shot]
print('接收器shape：', receiver_amplitudes_true.shape)
#  添加噪声
if AddNoise == True and noise_var != None:
    if noise_type == 'Gaussian':
        ResultPath = ResultPath + 'AWGN_var' + str(noise_var) + '/'
    if not os.path.exists(ResultPath):
        os.makedirs(ResultPath)
    noise_filepath = ResultPath + str(data_name) + '_noisercv_amps.mat'
    if os.path.exists(noise_filepath):
        print('Noisy receiver amplitudes exists, downloading......')
        # here the rcv_amps_true is contaminated by noise
        receiver_amplitudes_true = loadrcv(noise_filepath)

    else:
        print('正在生成噪声地震数据......\n',
              '添加 %.2f Gaussian噪声' % (noise_var))
        receiver_amplitudes_noise = add_noise(receiver_amplitudes_true, noise_type, noise_var)
        # 计算噪声平均SNR
        noise_snr = ComputeSNR(receiver_amplitudes_noise.permute(1, 0, 2),
                               receiver_amplitudes_true.permute(1, 0, 2))
        print('The noise level is %.2f dB' % noise_snr)
        # 打印噪声版数据
        plotoneshot(receiver_amplitudes_noise, ResultPath)
        # save the receiver amplitudes
        spio.savemat(noise_filepath,
                     {'true': receiver_amplitudes_noise.cpu().data.numpy(),
                      'rcv': receiver_amplitudes_true.cpu().data.numpy()})
print('数据生成成功~')
